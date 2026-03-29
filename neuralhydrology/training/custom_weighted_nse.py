import torch
from neuralhydrology.training.loss import BaseLoss

class CustomWeightedNSELoss(BaseLoss):
    def __init__(self, cfg, eps: float = 0.1, alpha: float = 2.0):
        super().__init__(
            cfg,
            prediction_keys=['y_hat'],
            ground_truth_keys=['y'],
            additional_data=['per_basin_target_stds']
        )
        self.eps = eps
        self.alpha = alpha

    def _get_loss(self, prediction, ground_truth, **kwargs):
        mask = ~torch.isnan(ground_truth['y'])

        y_hat = prediction['y_hat'][mask]
        y = ground_truth['y'][mask]

        std = kwargs['per_basin_target_stds']
        std = std.expand_as(prediction['y_hat'])[mask]

        base_weight = 1 / (std + self.eps) ** 2

        y_norm = (y - y.mean()) / (y.std() + 1e-6)

        extreme_weight = 1 + self.alpha * torch.relu(y_norm)

        loss = base_weight * extreme_weight * (y_hat - y) ** 2

        return torch.mean(loss)