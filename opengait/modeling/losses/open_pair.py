"""Open-pair loss: pairwise hinge between mated genuine distances and non-mated nearest-gallery distances.

    L_open-pair = 1 / (|P_M| |P_NM|) * sum_{x_m in P_M} sum_{x_u in P_NM}
                  max(0, d(x_m, g_{y(x_m)}) - min_{g in G} d(x_u, g) + m_open)

Every genuine distance has to sit at least m_open below the distance from every non-mated probe to its
nearest gallery template, so that a single global threshold separates accepted mated probes from rejected
intruders (a pairwise surrogate of the AUC between the two score distributions that set FNIR / FPIR).

Minibatch simulation of open-set evaluation, identical to OpenSetLoss (Su et al., ECCV 2024, open_set.py):
the subjects are split into num_folds folds; each fold in turn is the non-mated probe set P_NM, and every
other subject gives a gallery template g (mean of the first half of its sequences) and mated probes P_M
(second half). G is the set of templates of the fold. Distances: plain Euclidean on the raw (non-normalised)
part embeddings, per part, as in Su's OpenSetLoss / TripletLoss and their 'euc' test metric.

reduction:
  - 'nonzero' (default): mean over the active pairs (hinge > 0), as OpenGait's TripletLoss (AvgNonZeroReducer),
    so the loss does not fade out as most pairs satisfy the margin and its scale is comparable to the triplet.
  - 'mean': mean over all |P_M| |P_NM| pairs of the fold (the formula as written).
"""
import torch
from torch.nn import functional as F

from .base import BaseLoss, gather_and_scale_wrapper
from .open_set import compute_distance


class OpenPairLoss(BaseLoss):
    def __init__(self, num_folds=4, margin=0.2, reduction='nonzero', loss_term_weight=1.0):
        super(OpenPairLoss, self).__init__(loss_term_weight)
        if reduction not in ('nonzero', 'mean'):
            raise ValueError(f"reduction must be 'nonzero' or 'mean', got {reduction!r}")
        self.num_folds = num_folds
        self.margin = margin
        self.reduction = reduction

    @staticmethod
    def split_fold(labels, unique_labels, fold):
        """Su's partition: fold subjects -> non-mated probes; the rest -> mean template (first half) + mated probes."""
        gallery_idx, labels_gallery, probe_match, probe_nonmatch = [], [], [], []
        for subject in unique_labels:
            subject_indices = torch.nonzero(labels == subject, as_tuple=False).view(-1)
            if subject in fold:
                probe_nonmatch.extend(subject_indices.tolist())
            else:
                split_index = max(1, int(len(subject_indices) / 2 + .5))
                gallery_idx.append(subject_indices[:split_index])
                labels_gallery.append(subject)
                probe_match.extend(subject_indices[split_index:].tolist())
        return gallery_idx, labels_gallery, probe_match, probe_nonmatch

    @gather_and_scale_wrapper
    def forward(self, embeddings, labels):
        # embeddings: [n, c, p], labels: [n]
        unique_labels = torch.unique(labels)
        folds = torch.chunk(unique_labels, self.num_folds)

        hinges, d_gen_all, d_nm_all = [], [], []
        for fold in folds:
            gallery_idx, labels_gallery, probe_match, probe_nonmatch = self.split_fold(labels, unique_labels, fold)
            if len(probe_match) == 0 or len(probe_nonmatch) == 0:
                continue

            gallery = torch.stack([embeddings[idx].mean(0) for idx in gallery_idx])  # [n_g, c, p]
            gallery = gallery.permute(2, 0, 1).contiguous().float()  # [p, n_g, c]
            emb_m = embeddings[probe_match].permute(2, 0, 1).contiguous().float()  # [p, n_m, c]
            emb_u = embeddings[probe_nonmatch].permute(2, 0, 1).contiguous().float()  # [p, n_u, c]

            labels_gallery = torch.stack(labels_gallery)  # [n_g]
            match_mask = labels_gallery.unsqueeze(1) == labels[probe_match].unsqueeze(0)  # [n_g, n_m], one True per column

            dist_m = compute_distance(gallery, emb_m)  # [p, n_g, n_m]
            d_gen = (dist_m * match_mask.unsqueeze(0)).sum(1)  # [p, n_m]: d(x_m, g_y(x_m))
            d_nm = compute_distance(gallery, emb_u).min(1)[0]  # [p, n_u]: min_g d(x_u, g)

            hinges.append(F.relu(d_gen.unsqueeze(2) - d_nm.unsqueeze(1) + self.margin).flatten(1))  # [p, n_m * n_u]
            d_gen_all.append(d_gen.detach())
            d_nm_all.append(d_nm.detach())

        hinge = torch.cat(hinges, 1)  # [p, num_pairs over all folds]
        active = (hinge > 0).sum(1).float()  # [p]
        if self.reduction == 'nonzero':
            loss = hinge.sum(1) / (active + 1e-9)
            loss = torch.where(active > 0, loss, torch.zeros_like(loss))
        else:
            loss = hinge.mean(1)

        self.info.update({
            'loss': loss.detach().clone(),
            'active_frac': active / hinge.shape[1],
            'd_gen': torch.cat(d_gen_all, 1).mean(1),
            'd_nm_min': torch.cat(d_nm_all, 1).mean(1),
        })
        return loss, self.info
