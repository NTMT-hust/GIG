from torch import nn
import torch
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, num_classes, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # Standard Cross Entropy per sample
        ce_loss = F.cross_entropy(logits, targets, reduction='none')

        # pt = probability of true class
        pt = torch.exp(-ce_loss)

        # Focal scaling
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        # Class weighting (alpha)
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss