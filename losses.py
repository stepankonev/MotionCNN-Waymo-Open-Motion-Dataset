import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from abc import ABC


class Loss(ABC, nn.Module):
    def _precision_matrix(shape, sigma_xx, sigma_yy):
        assert sigma_xx.shape[-1] == 1
        assert sigma_xx.shape == sigma_yy.shape
        batch_size, n_modes, n_future_timstamps = \
            sigma_xx.shape[0], sigma_xx.shape[1], sigma_xx.shape[2]
        sigma_xx_inv = 1 / sigma_xx
        sigma_yy_inv = 1 / sigma_yy
        return torch.cat(
            [sigma_xx_inv, torch.zeros_like(sigma_xx_inv),
            torch.zeros_like(sigma_yy_inv), sigma_yy_inv], dim=-1) \
            .reshape(batch_size, n_modes, n_future_timstamps, 2, 2)

    def _log_N_conf(self, data_dict, prediction_dict):
        gt = data_dict['future_local'].unsqueeze(1)
        diff = (prediction_dict['xy'] - gt) * \
            data_dict['future_valid'][:, None, :, None]
        assert torch.isfinite(diff).all()
        precision_matrices = self._precision_matrix(
            prediction_dict['sigma_xx'], prediction_dict['sigma_yy'])
        assert torch.isfinite(precision_matrices).all()
        log_confidences = torch.log_softmax(
            prediction_dict['confidences'], dim=-1)
        assert torch.isfinite(log_confidences).all()
        bilinear = diff.unsqueeze(-2) @ precision_matrices @ diff.unsqueeze(-1)
        bilinear = bilinear[:, :, :, 0, 0]
        assert torch.isfinite(bilinear).all()
        log_N = -0.5 * np.log(2 * np.pi) - 0.5 * torch.log(
            prediction_dict['sigma_xx'] * prediction_dict['sigma_yy']
            ).squeeze(-1) - 0.5 * bilinear
        return log_N, log_confidences


class NLLGaussian2d(Loss):
    def __init__(self):
        super().__init__()

    def forward(self, data_dict, prediction_dict):
        log_N, log_confidences = self._log_N_conf(data_dict, prediction_dict)
        assert torch.isfinite(log_N).all()
        log_L = torch.logsumexp(log_N.sum(dim=2) + log_confidences, dim=1)
        assert torch.isfinite(log_L).all()
        return -log_L.mean()
