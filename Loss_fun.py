import torch
import torch.nn as nn
import torch.nn.functional as F

from loss.softmax_loss import CrossEntropyLabelSmooth
from loss.triplet_loss import TripletLoss
from loss.center_loss import CenterLoss


class SupConLoss(nn.Module):
    """Supervised contrastive loss over a single batch.

    Positives: same identity in the batch (excluding self).
    Negatives: different identities in the batch.
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

        device = features.device
        batch_size = features.size(0)
        if batch_size <= 1:
            return features.sum() * 0.0

        features = F.normalize(features, p=2, dim=1)
        labels = labels.contiguous().view(-1, 1)

        mask = torch.eq(labels, labels.t()).float().to(device)
        logits = torch.div(torch.matmul(features, features.t()), self.temperature)
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
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)
    center_criterion_local = CenterLoss(num_classes=num_classes, feat_dim=3072, use_gpu=True)

    triplet = TripletLoss()
    xent = CrossEntropyLabelSmooth(num_classes=num_classes)
    supcon = SupConLoss(temperature=contrast_temp)

    def loss_func(score, feat, target, contrast_feat=None):
        if isinstance(score, list):
            id_loss = [xent(scor, target) for scor in score[1:]]
            id_loss = sum(id_loss) / len(id_loss)
            id_loss = 0.25 * id_loss + 0.75 * xent(score[0], target)
        else:
            id_loss = xent(score, target)

        if isinstance(feat, list):
            tri_loss = [triplet(feats, target)[0] for feats in feat[1:]]
            tri_loss = sum(tri_loss) / len(tri_loss)
            tri_loss = 0.25 * tri_loss + 0.75 * triplet(feat[0], target)[0]

            center_global = center_criterion(feat[0], target)
            center_locals = [center_criterion_local(feats, target) for feats in feat[1:]]
            center_locals = sum(center_locals) / len(center_locals)
            center_loss = 0.25 * center_locals + 0.75 * center_global
        else:
            tri_loss = triplet(feat, target)[0]
            center_loss = center_criterion(feat, target)

        contrast_loss = feat[0].sum() * 0.0 if isinstance(feat, list) else feat.sum() * 0.0
        if contrast_feat is not None:
            contrast_loss = supcon(contrast_feat, target)

        return id_loss + tri_loss, center_loss, contrast_loss

    return loss_func, center_criterion
