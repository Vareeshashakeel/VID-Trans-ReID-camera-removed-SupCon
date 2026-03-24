import torch
import torch.nn as nn
import torch.nn.functional as F

from loss.softmax_loss import CrossEntropyLabelSmooth
from loss.triplet_loss import TripletLoss
from loss.center_loss import CenterLoss


class SupConLoss(nn.Module):
    """
    Supervised contrastive loss on one batch.

    positives = same identity in batch (excluding self)
    negatives = different identities in batch
    """

    def __init__(self, temperature=0.07, eps=1e-12):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, features, labels):
        if features.dim() != 2:
            raise ValueError(f"SupConLoss expects [B, D], got {tuple(features.shape)}")

        if labels.dim() != 1:
            labels = labels.view(-1)

        batch_size = features.size(0)
        if batch_size <= 1:
            return features.sum() * 0.0

        device = features.device
        features = F.normalize(features, p=2, dim=1)
        labels = labels.contiguous().view(-1, 1)

        mask = torch.eq(labels, labels.t()).float().to(device)

        logits = torch.matmul(features, features.t()) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True)[0].detach()

        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + self.eps)

        positive_counts = mask.sum(dim=1)
        valid = positive_counts > 0

        if not torch.any(valid):
            return features.sum() * 0.0

        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (positive_counts + self.eps)
        loss = -mean_log_prob_pos[valid].mean()
        return loss


def make_loss(num_classes, contrast_temp=0.07):
    feat_dim = 768

    center_criterion = CenterLoss(
        num_classes=num_classes,
        feat_dim=feat_dim,
        use_gpu=True
    )

    triplet = TripletLoss()
    xent = CrossEntropyLabelSmooth(num_classes=num_classes)
    supcon = SupConLoss(temperature=contrast_temp)

    def loss_func(score, feat, target, contrast_feat=None):
        # ID loss
        if isinstance(score, list):
            local_id_loss = [xent(scor, target) for scor in score[1:]]
            local_id_loss = sum(local_id_loss) / len(local_id_loss)
            id_loss = 0.25 * local_id_loss + 0.75 * xent(score[0], target)
        else:
            id_loss = xent(score, target)

        # Triplet + center
        if isinstance(feat, list):
            local_tri_loss = [triplet(f, target)[0] for f in feat[1:]]
            local_tri_loss = sum(local_tri_loss) / len(local_tri_loss)
            tri_loss = 0.25 * local_tri_loss + 0.75 * triplet(feat[0], target)[0]

            # center loss only on global feature
            center_loss = center_criterion(feat[0], target)
        else:
            tri_loss = triplet(feat, target)[0]
            center_loss = center_criterion(feat, target)

        # supervised contrastive loss on global feature
        if contrast_feat is not None:
            contrast_loss = supcon(contrast_feat, target)
        else:
            if isinstance(feat, list):
                contrast_loss = feat[0].sum() * 0.0
            else:
                contrast_loss = feat.sum() * 0.0

        return id_loss + tri_loss, center_loss, contrast_loss

    return loss_func, center_criterion
