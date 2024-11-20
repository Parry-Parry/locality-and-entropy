from rankers.train.loss import BaseLoss, register_loss
from torch import Tensor
import torch
import torch.nn.functional as F


@register_loss("onesided_margin_mse")
class OneSidedMarginMSELoss(BaseLoss):
    """Margin MSE loss with residual calculation."""

    def forward(self, pred: Tensor, labels: Tensor) -> Tensor:
        # norm pred and labels
        pred = pred / pred.norm(dim=1, keepdim=True)
        labels = labels / labels.norm(dim=1, keepdim=True)

        residual_pred = pred[:, 0].unsqueeze(1) - pred[:, 1:]
        residual_label = torch.ones_like(labels[:, 0].unsqueeze(1)) - labels[:, 1:]
        return F.mse_loss(residual_pred, residual_label, reduction=self.reduction)
