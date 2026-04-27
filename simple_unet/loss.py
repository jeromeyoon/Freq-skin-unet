import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, gt):
        # pred, gt: [B, 1, H, W], values in [0,1]
        bce = F.binary_cross_entropy(pred, gt, reduction='none')
        pt  = torch.where(gt == 1, pred, 1 - pred)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, gt):
        pred_flat = pred.view(pred.size(0), -1)
        gt_flat   = gt.view(gt.size(0), -1)
        intersection = (pred_flat * gt_flat).sum(dim=1)
        dice = (2 * intersection + self.smooth) / (pred_flat.sum(dim=1) + gt_flat.sum(dim=1) + self.smooth)
        return (1 - dice).mean()


class SegLoss(nn.Module):
    def __init__(self, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice  = DiceLoss()

    def forward(self, pred, gt):
        return self.focal(pred, gt) + self.dice(pred, gt)
