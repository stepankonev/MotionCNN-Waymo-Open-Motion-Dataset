import numpy as np
import torch
import timm
import os
import argparse
from glob import glob
from tqdm import tqdm
from utils import get_config
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from losses import NLLGaussian2d


class MotionCNNDataset(Dataset):
    def __init__(self, data_path, load_roadgraph=False) -> None:
        super().__init__()
        self._load_roadgraph = load_roadgraph
        self._files = glob(os.path.join(data_path, '*', 'agent_data', '*.npz'))
        self._roadgraph_data = glob(os.path.join(
            data_path, '*', 'roadgraph_data', 'segments_global.npz'))
        self._scid_to_roadgraph = {
            f.split('/')[-3]: f for f in self._roadgraph_data}

    def __len__(self):
        return len(self._files)

    def __getitem__(self, idx):
        data = dict(np.load(self._files[idx], allow_pickle=True))
        if self._load_roadgraph:
            roadgraph_data_file = \
                self._scid_to_roadgraph[data['scenario_id'].item()]
            roadgraph_data = np.load(roadgraph_data_file)['roadgraph_segments']
            roadgraph_valid = np.ones(roadgraph_data.shape[0])
            n_to_pad = 6000 - roadgraph_data.shape[0]
            roadgraph_data = np.pad(
                roadgraph_data, ((0, n_to_pad), (0, 0), (0, 0)))
            roadgraph_valid = np.pad(roadgraph_valid, (0, n_to_pad))
            data['roadgraph_data'] = roadgraph_data
            data['roadgraph_valid'] = roadgraph_valid
        data['raster'] = data['raster'].transpose(2, 0, 1) / 255.
        data['scenario_id'] = data['scenario_id'].item()
        return data


def dict_to_cuda(data_dict):
    gpu_required_keys = ['raster', 'future_valid', 'future_local']
    for key in gpu_required_keys:
        data_dict[key] = data_dict[key].cuda()
    return data_dict


def get_model(model_config):
    # x, y, sigma_xx, sigma_yy, visibility
    n_components = 5
    n_modes = model_config['n_modes']
    n_timestamps = model_config['n_timestamps']
    output_dim = n_modes + n_modes * n_timestamps * n_components
    model = timm.create_model(
        model_config['backbone'], pretrained=True,
        in_chans=27, num_classes=output_dim)
    return model


def limited_softplus(x):
    return torch.clamp(F.softplus(x), min=0.1, max=10)


def postprocess_predictions(predicted_tensor, model_config):
    confidences = predicted_tensor[:, :model_config['n_modes']]
    components = predicted_tensor[:, model_config['n_modes']:]
    components = components.reshape(
        -1, model_config['n_modes'], model_config['n_timestamps'], 5)
    sigma_xx = components[:, :, :, 2:3]
    sigma_yy = components[:, :, :, 3:4]
    visibility = components[:, :, :, 4:]
    return {
        'confidences': confidences,
        'xy': components[:, :, :, :2],
        'sigma_xx': limited_softplus(sigma_xx) if \
            model_config['predict_covariances'] else torch.ones_like(sigma_xx),
        'sigma_yy': limited_softplus(sigma_yy) if \
            model_config['predict_covariances'] else torch.ones_like(sigma_yy),
        'visibility': visibility}


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-data-path", type=str, required=True,
        help="Path to training data")
    parser.add_argument(
        "--val-data-path", type=str, required=True,
        help="Path to validation data")
    parser.add_argument(
        "--checkpoints-path", type=str, required=True,
        help="Path to checkpoints")
    parser.add_argument(
        "--config", type=str, required=True, help="Config file path")
    parser.add_argument("--multi-gpu", action='store_true')
    args = parser.parse_args()
    return args


def get_last_checkpoint_file(path):
    list_of_files = glob(f'{path}/*.pth')
    if len(list_of_files) == 0:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


def main():
    args = parse_arguments()
    general_config = get_config(args.config)
    model_config = general_config['model']
    training_config = general_config['training']
    config_name = args.config.split('/')[-1].split('.')[0]
    model = get_model(model_config)
    model.cuda()
    optimizer = Adam(model.parameters(), **training_config['optimizer'])
    loss_module = NLLGaussian2d()
    processed_batches = 0
    epochs_processed = 0
    train_losses = []
    experiment_checkpoints_dir = os.path.join(
        args.checkpoints_path, config_name)
    if not os.path.exists(experiment_checkpoints_dir):
        os.makedirs(experiment_checkpoints_dir)
    latest_checkpoint = get_last_checkpoint_file(experiment_checkpoints_dir)
    if latest_checkpoint is not None:
        print(f"Loading checkpoint from {latest_checkpoint}")
        checkpoint_data = torch.load(latest_checkpoint)
        model.load_state_dict(checkpoint_data['model_state_dict'])
        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        epochs_processed = checkpoint_data['epochs_processed']
        processed_batches = checkpoint_data['processed_batches']
    if args.multi_gpu:
        model = nn.DataParallel(model)
    training_dataloader = DataLoader(
        MotionCNNDataset(args.train_data_path),
        **training_config['train_dataloader'])
    validation_dataloader = DataLoader(
        MotionCNNDataset(args.val_data_path, load_roadgraph=True),
        **training_config['val_dataloader'])

    for epochs_processed in tqdm(
            range(epochs_processed, training_config['num_epochs']),
            total=training_config['num_epochs'],
            initial=epochs_processed):
        train_progress_bar = tqdm(
            training_dataloader, total=len(training_dataloader))
        for train_data in train_progress_bar:
            optimizer.zero_grad()
            train_data = dict_to_cuda(train_data)
            prediction_tensor = model(train_data['raster'].float())
            prediction_dict = postprocess_predictions(
                prediction_tensor, model_config)
            loss = loss_module(train_data, prediction_dict)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            processed_batches += 1
            train_progress_bar.set_description(
                "Train loss: %.3f" % np.mean(train_losses[-100:]))
            if processed_batches % training_config['eval_every'] == 0:
                del train_data
                torch.cuda.empty_cache()
                with torch.no_grad():
                    for eval_data in tqdm(validation_dataloader):
                        eval_data = dict_to_cuda(eval_data)
                        prediction_tensor = model(eval_data['raster'].float())
                        prediction_dict = \
                            postprocess_predictions(
                            prediction_tensor, model_config)
                        loss = loss_module(eval_data, prediction_dict)
                if  isinstance(model, nn.DataParallel):
                    model_state_dict = model.module.state_dict()
                else:
                    model_state_dict = model.state_dict()
                torch_checkpoint_data = {
                    "model_state_dict": model_state_dict,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epochs_processed": epochs_processed,
                    "processed_batches": processed_batches}
                torch_checkpoint_path = os.path.join(
                    experiment_checkpoints_dir,
                    f'e{epochs_processed}_b{processed_batches}.pth')
                torch.save(torch_checkpoint_data, torch_checkpoint_path)


if __name__ == '__main__':
    main()
