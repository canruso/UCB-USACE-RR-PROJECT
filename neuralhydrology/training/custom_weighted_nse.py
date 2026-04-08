from typing import Dict
import torch
from neuralhydrology.training.loss import BaseLoss

class CustomWeightedNSELoss(BaseLoss):
    def __init__(self, cfg, eps: float = 0.1, alpha: float = 2.0, peak_penalty: float = 10000000.0):
        super(CustomWeightedNSELoss, self).__init__(
            cfg,
            prediction_keys=['y_hat'],
            ground_truth_keys=['y'],
            additional_data=['per_basin_target_stds', 'is_peak']
        )
        self.eps = eps
        self.alpha = alpha
        self.peak_penalty = peak_penalty
        self.debug_plot = False

    def _get_loss(self, prediction: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor], **kwargs):
        mask = ~torch.isnan(ground_truth['y'])

        y_hat = prediction['y_hat'][mask]
        y = ground_truth['y'][mask]

        per_basin_target_stds = kwargs['per_basin_target_stds']
        per_basin_target_stds = per_basin_target_stds.expand_as(prediction['y_hat'])[mask]

        if 'is_peak' in kwargs:
            is_peak = kwargs['is_peak'][:, -prediction['y_hat'].shape[1]:, :]
            is_peak = is_peak.expand_as(prediction['y_hat'])[mask]
        else:
            is_peak = torch.zeros_like(prediction['y_hat'])[mask]

        base_weights = 1 / (per_basin_target_stds + self.eps) ** 2
        peak_weights = 1 + self.peak_penalty * is_peak
        weights = base_weights * peak_weights

        squared_error = (y_hat - y) ** 2
        scaled_loss = weights * squared_error

        if self.debug_plot:
            import matplotlib.pyplot as plt
            import numpy as np

            y_full = ground_truth['y']

            if 'is_peak' in kwargs:
                peak_full = kwargs['is_peak'][:, -prediction['y_hat'].shape[1]:, :]
            else:
                peak_full = torch.zeros_like(ground_truth['y'])

            if isinstance(y_full, torch.Tensor):
                y_full = y_full.detach().cpu().numpy()
            if isinstance(peak_full, torch.Tensor):
                peak_full = peak_full.detach().cpu().numpy()

            y_full = np.squeeze(y_full)
            peak_full = np.squeeze(peak_full)

            if y_full.ndim == 1:
                y_full = y_full[None, :]
                peak_full = peak_full[None, :]

            B, T = y_full.shape

            for b in range(B):
                y_seq = y_full[b]
                p_seq = peak_full[b]

                valid_mask = ~np.isnan(y_seq)
                if valid_mask.sum() == 0:
                    continue

                plt.figure(figsize=(14, 4))
                plt.plot(np.where(valid_mask)[0], y_seq[valid_mask], lw=0.8)

                in_peak = False
                for t in range(T):
                    if p_seq[t] and not in_peak:
                        s = t
                        in_peak = True
                    elif not p_seq[t] and in_peak:
                        plt.axvspan(s, t - 1, color='red', alpha=0.2)
                        in_peak = False

                if in_peak:
                    plt.axvspan(s, T - 1, color='red', alpha=0.2)

                plt.title(f"Batch {b} Peak Regions")
                plt.tight_layout()
                plt.show()

        return torch.mean(scaled_loss)

    @staticmethod
    def _subset_additional_data(additional_data: Dict[str, torch.Tensor], n_target: int) -> Dict[str, torch.Tensor]:
        out = {}
        for key, value in additional_data.items():
            if key == 'is_peak':
                out[key] = value
            else:
                out[key] = value[:, :, n_target:n_target + 1]
        return out