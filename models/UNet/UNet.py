from monai.networks.nets import UNet as MonaiUNet
import torch.nn as nn
import torch
import torch.nn.functional as F

class UNet(nn.Module):
    """
    A wrapper for MONAI's UNet.
    Training loss (Soft Dice + BCE) is encapsulated within the model.
    """
    def __init__(
        self, 
        spatial_dims, 
        in_channels, 
        out_channels, 
        channels=(32, 64, 128, 256, 512), 
        strides=(2, 2, 2, 2),
        num_res_units=2,
        dropout=0.2
    ):
        super().__init__()
        self.model = MonaiUNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            dropout=dropout
        )

    def forward(self, x):
        return self.model(x)

    def get_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculates the hybrid Dice + BCE loss.
        """
        # 1. BCE Loss
        # target is cast to float to match pred; from_logits=True is assumed from trainer
        bce = F.binary_cross_entropy_with_logits(pred, target.float())

        # 2. Soft Dice Loss
        pred_soft = pred.sigmoid()
        target_soft = target.float()
        
        # Sum over all dimensions except batch
        dims = tuple(range(1, pred_soft.dim()))
        intersection = (pred_soft * target_soft).sum(dim=dims)
        union = pred_soft.sum(dim=dims) + target_soft.sum(dim=dims)
        
        smooth = 1e-5
        dice_score = (2.0 * intersection + smooth) / (union + smooth)
        dice_loss = (1.0 - dice_score).mean()

        # Weighted combination (50/50)
        return 0.5 * bce + 0.5 * dice_loss