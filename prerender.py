import numpy as np
import cv2
import os
import json
import argparse
import multiprocessing
import tensorflow as tf
from tqdm import tqdm
from utils import create_tf_dataset, get_config
from features_description import get_features_description
from extra_math import shift_rotate, rotate_shift, rot_matrix


class RoadgraphProcessor:
    def __init__(self, data, config):
        self._config = config
        self._segments = None
        validity_flag = data['roadgraph_samples/valid'].numpy().flatten()
        self._roadgraph_xy = data['roadgraph_samples/xyz'] \
            .numpy()[validity_flag == 1][:, :2]
        self._roadgraph_type = data['roadgraph_samples/type'] \
            .numpy().flatten()[validity_flag == 1]
        self._ids = data['roadgraph_samples/id'] \
            .numpy().flatten()[validity_flag == 1]

    def _get_splits(self):
        splits = []
        prev_value = self._ids[0]
        for i, idx in enumerate(self._ids):
            if idx != prev_value:
                splits.append(i)
                prev_value = idx
        splits.append(len(self._ids))
        splits = \
            [[splits[i - 1], splits[i]] for i in range(1, len(splits))]
        return splits

    def _get_color(self, segment_type):
        type_to_color = {
            1:  (0, 0, 0, 0),
            2:  (0, 0, 0, 255),
            3:  (0, 0, 255, 0),
            6:  (0, 0, 255, 255),
            7:  (0, 255, 0, 0),
            8:  (0, 255, 0, 255),
            9:  (0, 255, 255, 0),
            10: (0, 255, 255, 255),
            11: (255, 0, 0, 0),
            12: (255, 0, 0, 255),
            13: (255, 0, 255, 0),
            15: (255, 0, 255, 255),
            16: (255, 255, 0, 0),
            17: (255, 255, 0, 255),
            18: (255, 255, 255, 0),
            19: (255, 255, 255, 255)}
        return type_to_color[segment_type]

    def _prepare_segments(self):
        graph_segments = []
        graph_types = []
        splits = self._get_splits()
        for (start, end) in splits:
            num_points = max(
                int((end - start) \
                     / self._config['roadgraph_distillation_rate']), 2)
            roadline_ids = self._ids[start:end]
            roadline_types = self._roadgraph_type[start:end]
            assert all(roadline_ids == roadline_ids[0])
            if roadline_types[0] == 18:
                distilled_roadline_data = self._roadgraph_xy[start:end]
            else:
                idx = np.linspace(start, end - 1, num_points).astype(int)
                distilled_roadline_data = self._roadgraph_xy[idx]
            segments = np.concatenate([
                np.pad(distilled_roadline_data, ((0, 1), (0, 0)))[:, None, :],
                np.pad(distilled_roadline_data, ((1, 0), (0, 0)))[:, None, :]],
                axis=1)[1:-1]
            graph_segments.append(segments)
            graph_types.extend([roadline_types[0]] * segments.shape[0])
        self._segments = np.concatenate(graph_segments, axis=0)
        self._segment_types = graph_types

    def center_to(self, target_agent_xy, target_agent_yaw):
        return shift_rotate(
            self._segments, -target_agent_xy, -target_agent_yaw)

    def json(self):
        return json.dumps(np.round(self._segments.tolist(), 2).tolist())

    def __str__(self) -> str:
        return self.json()

    def render(self, target_agent_xy, target_agent_yaw):
        if self._segments is None:
            self._prepare_segments()
        segments = self.center_to(target_agent_xy, target_agent_yaw)
        masked_raster = np.zeros((
            self._config['raster_size'], self._config['raster_size'], 1),
            np.uint8)
        typed_raster = np.zeros((
            self._config['raster_size'], self._config['raster_size'], 4),
            np.uint8)
        for segment_type, segment in zip(self._segment_types, segments):
            int_segment = (segment * self._config['scale'] + \
                np.array(
                    [self._config['center_x'], self._config['center_y']])) \
                    .astype(int)
            masked_raster = cv2.line(
                masked_raster,
                int_segment[0], int_segment[1],
                255, 1)
            typed_raster = cv2.line(
                typed_raster,
                int_segment[0], int_segment[1],
                self._get_color(segment_type), 1)
        raster = np.concatenate([masked_raster, typed_raster], axis=-1)
        return raster
    
    def get_roadgraph_segments_data(self):
        return {'roadgraph_segments': self._segments}


class AgentProcessor:
    def __init__(self, data, config):
        self._config = config
        # currently_visible = data['state/current/valid'].numpy().flatten()
        history_valid = np.concatenate([
            data['state/past/valid'].numpy(),
            data['state/current/valid'].numpy()], axis=-1)
        assert history_valid.shape[1] == 11
        present_in_history = np.max(history_valid, axis=-1)
        self._is_target = data['state/tracks_to_predict'].numpy().flatten()
        selector = np.logical_or(present_in_history == 1, self._is_target == 1)
        self._history_xy = np.concatenate([
            np.concatenate([
                data['state/past/x'].numpy(),
                data['state/current/x'].numpy()], axis=-1)[:, :, None],
            np.concatenate([
                data['state/past/y'].numpy(),
                data['state/current/y'].numpy()], axis=-1)[:, :, None]],
            axis=-1)[selector]
        self._history_yaw = np.concatenate([
            data['state/past/bbox_yaw'].numpy(),
            data['state/current/bbox_yaw'].numpy()],
            axis=-1)[selector]
        self._history_valid = np.concatenate([
            data['state/past/valid'].numpy(),
            data['state/current/valid'].numpy()],
            axis=-1)[selector]
        self._future_xy = np.concatenate([
            data['state/future/x'].numpy()[:, :, None],
            data['state/future/y'].numpy()[:, :, None]],
            axis=-1)[selector]
        self._future_valid = \
            data['state/future/valid'].numpy()[selector]
        self._current_xy = np.concatenate([
            data['state/current/x'].numpy(),
            data['state/current/y'].numpy()], axis=-1)[selector]
        self._history_speed = data['state/past/speed'] \
            .numpy()[selector]
        self._current_speed = data['state/current/speed'] \
            .numpy().flatten()[selector]
        self._future_speed = data['state/future/speed'] \
            .numpy()[selector]
        self._current_yaw = \
            data['state/current/bbox_yaw'].numpy().flatten()\
                [selector]
        self._agents_id = \
            data['state/id'].numpy().flatten() \
                .astype(int)[selector]
        self._is_sdc = \
            data['state/is_sdc'].numpy().flatten() \
                .astype(int)[selector]
        self._scenario_id = \
            data['scenario/id'].numpy().item().decode()
        self._agents_type = \
            data['state/type'].numpy().flatten() \
                .astype(int)[selector]
        self._agents_width = \
            data['state/current/width'] \
                .numpy().flatten()[selector]
        self._agents_length = \
            data['state/current/length'] \
                .numpy().flatten()[selector]

    def target_agents_idx(self):
        return np.arange(len(self._is_target))[self._is_target == 1]

    def get_target_agent_position(self, idx):
        return self._current_xy[idx], self._current_yaw[idx]

    def _gen_box(
            self, current_agent_xy, current_agent_yaw,
            target_agent_xy, target_agent_yaw,
            current_agent_length, current_agent_width):
        box = np.array([
            [-current_agent_length / 2, -current_agent_width / 2],
            [ current_agent_length / 2, -current_agent_width / 2],
            [ current_agent_length / 2,  current_agent_width / 2],
            [-current_agent_length / 2,  current_agent_width / 2]])[None, ]
        box *= self._config['scale']
        box = box @ rot_matrix(current_agent_yaw).T
        box = shift_rotate(
            box, (current_agent_xy - target_agent_xy) * self._config['scale'],
            -target_agent_yaw)
        return box
    
    def _draw_box(self, raster,
            current_agent_xy, current_agent_yaw,
            target_agent_xy, target_agent_yaw,
            current_agent_length, current_agent_width):
        raster = cv2.fillPoly(
            raster, 
            (self._gen_box(
                current_agent_xy, current_agent_yaw,
                target_agent_xy, target_agent_yaw,
                current_agent_length, current_agent_width) + \
                    np.array([
                        self._config['center_x'], self._config['center_y']])) \
                        .astype(int),
            128, lineType=cv2.LINE_AA)
        raster = cv2.polylines(
            raster, 
            (self._gen_box(
                current_agent_xy, current_agent_yaw,
                target_agent_xy, target_agent_yaw,
                current_agent_length, current_agent_width) + \
                    np.array([
                        self._config['center_x'], self._config['center_y']])) \
                        .astype(int),
            True, 255, lineType=cv2.LINE_AA, thickness=1)
        return raster
    
    def render(self, target_agent_order_idx):
        agents_raster = [np.zeros((
            self._config['raster_size'], self._config['raster_size'], 1),
                np.uint8) for _ in range(22)]
        target_agent_xy, target_agent_yaw = \
            self.get_target_agent_position(target_agent_order_idx)
        for rendering_agent_agent_order_idx, (
                    rendering_agent_history_xy,
                    rendering_agent_history_yaw, 
                    rendering_agent_length,
                    rendering_agent_width,
                    rendering_agent_history_valid) in enumerate(zip(
                self._history_xy, self._history_yaw,
                self._agents_length, self._agents_width, self._history_valid)):
            for history_timestamp, (rendering_agent_history_xy_state,
                                    rendering_agent_history_yaw_state,
                                    rendering_agent_history_valid_state) in \
                    enumerate(zip(
                        rendering_agent_history_xy,
                        rendering_agent_history_yaw,
                        rendering_agent_history_valid)):
                if rendering_agent_history_valid_state == 0:
                    continue
                channel = history_timestamp
                if target_agent_order_idx == rendering_agent_agent_order_idx:
                    channel += 11
                agents_raster[channel] = self._draw_box(
                    agents_raster[channel],
                    rendering_agent_history_xy_state,
                    rendering_agent_history_yaw_state,
                    target_agent_xy, target_agent_yaw,
                    rendering_agent_length,
                    rendering_agent_width)
        agents_raster = np.concatenate(agents_raster, axis=-1)
        return agents_raster
    
    def get_numerical_data(self, agent_order_idx):
        agent_gt_global = self._future_xy[agent_order_idx]
        target_agent_shift, target_agent_yaw = \
            self.get_target_agent_position(agent_order_idx)
        numerical_data = {
            'agent_id': self._agents_id[agent_order_idx],
            'scenario_id': self._scenario_id,
            'is_sdc': self._is_sdc[agent_order_idx],
            'agent_type': self._agents_type[agent_order_idx],
            'future_global': agent_gt_global,
            'future_local': shift_rotate(
                agent_gt_global, -target_agent_shift, -target_agent_yaw),
            'future_valid': self._future_valid[agent_order_idx],
            'history_global': self._history_xy[agent_order_idx],
            'history_valid': self._history_valid[agent_order_idx],
            'history_yaw_global': self._history_yaw[agent_order_idx],
            'current_xy_global': self._current_xy[agent_order_idx],
            'history_speed': self._history_speed[agent_order_idx],
            'current_speed': self._current_speed[agent_order_idx],
            'future_speed': self._future_speed[agent_order_idx],
            'width': self._agents_width[agent_order_idx],
            'length': self._agents_length[agent_order_idx],
            'shift': target_agent_shift,
            'yaw': target_agent_yaw}
        return numerical_data
        

def generate_filename(data_dict):
    return str(data_dict['agent_id']) + '.npz'


def create_folder_if_not_exisits(path):
    if not os.path.exists(path):
        os.makedirs(path)


def create_saving_paths(path, data_dict):
    scenario_folder = os.path.join(path, data_dict['scenario_id'])
    agent_data_folder = os.path.join(scenario_folder, 'agent_data')
    roadgraph_data_folder = os.path.join(scenario_folder, 'roadgraph_data')
    create_folder_if_not_exisits(scenario_folder)
    create_folder_if_not_exisits(agent_data_folder)
    create_folder_if_not_exisits(roadgraph_data_folder)


def save_agent_data(path, data_dict):
    scenario_folder = os.path.join(path, data_dict['scenario_id'])
    agent_data_folder = os.path.join(scenario_folder, 'agent_data')
    create_saving_paths(path, data_dict)
    np.savez_compressed(
        os.path.join(agent_data_folder, generate_filename(data_dict)),
        **data_dict)


def save_roadgraph_data(path, data_dict, roadgraph_data):
    scenario_folder = os.path.join(path, data_dict['scenario_id'])
    roadgraph_data_folder = os.path.join(scenario_folder, 'roadgraph_data')
    create_saving_paths(path, data_dict)
    np.savez_compressed(
        os.path.join(roadgraph_data_folder, 'segments_global.npz'),
        **roadgraph_data)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path", type=str, required=True, help="Path to raw data")
    parser.add_argument(
        "--output-path", type=str, required=True, help="Path to save data")
    parser.add_argument(
        "--n-jobs", type=int, default=1, required=False,
        help="Number of threads")
    parser.add_argument(
        "--n-shards", type=int, default=1, required=False,
        help="Use `1/n_shards` of full dataset")
    parser.add_argument(
        "--shard-id", type=int, default=0, required=False,
        help="Take shard with given id")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Config file path")
    args = parser.parse_args()
    return args


def process_and_save(data, config, output_path):
    data = tf.io.parse_single_example(data, get_features_description())
    agent_processor = AgentProcessor(data, config)
    roadgraph_processor = RoadgraphProcessor(data, config)
    for i in agent_processor.target_agents_idx():
        agents_raster = agent_processor.render(i)
        roadgraph_raster = roadgraph_processor.render(
            *agent_processor.get_target_agent_position(i))
        full_raster = np.concatenate(
            [roadgraph_raster, agents_raster], axis=-1)
        prepared_data = {'raster': full_raster}
        prepared_data.update(agent_processor.get_numerical_data(i))
        save_agent_data(output_path, prepared_data)
    save_roadgraph_data(
        output_path, prepared_data,
        roadgraph_processor.get_roadgraph_segments_data())


def main():
    args = parse_arguments()
    dataset = create_tf_dataset(args.data_path, args.n_shards, args.shard_id)
    config = get_config(args.config)['prerender']

    p = multiprocessing.Pool(args.n_jobs)
    processes = []
    for data in tqdm(dataset.as_numpy_iterator()):
        processes.append(
            p.apply_async(
                process_and_save,
                kwds=dict(
                    data=data,
                    config=config,
                    output_path=args.output_path)))
    for r in tqdm(processes):
        r.get()


if __name__ == '__main__':
    main()
