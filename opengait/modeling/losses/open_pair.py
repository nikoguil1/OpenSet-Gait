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

memory_bank (optional): the in-batch gallery has ~24 templates, the test gallery ~800, so the in-batch
min_g d(x_u, g) overestimates the distance to the nearest intruder (Gait3D, GaitBase at 60k: 1.37 with 24
templates vs 1.09 with all 2999 training identities; 1.10 on the test set). With memory_bank: true,
G = in-batch templates U {b_j : j != y(x_u), j seen}, where b_j is a template of every training identity:
the mean of its embed_1 in the batch, kept as an exponential moving average over the batches where it appears
(b_j <- bank_momentum * b_j + (1 - bank_momentum) * mean). Same kind of template and same distance as the
genuine term. The bank is a constant (no gradient), it is updated after the loss with the current batch, and
it is not saved in the checkpoint (rebuilt in ~K / subjects-per-batch iterations after a restart). The genuine
term d(x_m, g_y(x_m)) still uses the in-batch template. Labels must be class indices in [0, num_classes).

scale_reference (optional, idea 1 of the post-bank analysis): with raw distances and an absolute margin the model
can satisfy the margin by stretching / shrinking the embedding space instead of separating the classes. With
scale_reference='genuine', every distance of a fold is divided, per part, by s = mean_{x_m} d(x_m, g_y(x_m)) (the
mean genuine distance of that fold), as OpenSetLoss with scale_invariant + scale_reference='genuine' (E4.6):
    max(0, d_gen / s - d_nm / s + m_open)
so m_open is a relative margin. scale_detach=False (default) lets the gradient flow through s: the loss is then
exactly invariant to the embedding scale and has no radial gradient. scale_reference='none' (default) keeps the raw
distances. The logged d_gen / d_nm_min stay raw; 'scale' logs s.

tail_frac (optional, idea 2): FNIR@1%FPIR is set by the few intruders closest to the gallery, while the mean over all
(x_m, x_u) pairs is a surrogate of the full AUC (in our runs AUC / OSCR / FNIR@5% kept improving after 40k while
FNIR@1% got worse). With tail_frac = beta in (0, 1), each fold keeps, per part, only the ceil(beta * |P_NM|)
non-mated probes with the smallest d_nm (the hardest intruders) and the hinge runs over P_M x (that tail): the
standard surrogate of the one-way partial AUC over FPR in [0, beta] (Narasimhan & Agarwal, ICML 2013).
tail_frac=None (default) uses every non-mated probe.

nm_sequences (optional, idea 3a): the in-batch min over ~24 templates overestimates the test min over ~785
(extreme-value bias, see memory_bank above). With nm_sequences=True the candidates for the nearest gallery item of a
non-mated probe are the templates of the fold AND every individual sequence of the known (non-fold) subjects of the
batch (template and mated-probe sequences alike, ~4x more candidates), all computed with the current model and with
gradient, so nothing is stale (unlike the bank):
    d_nm(x_u) = min( min_{g in G} d(x_u, g), min_{x in S_K} d(x_u, x) )
S_K holds only subjects outside the fold, so the probe's own identity is never a candidate. The genuine term keeps
the in-batch template. Logs 'd_nm_min_tmpl' (templates only) and 'nm_from_seq' (fraction of non-mated probes whose
min is a sequence). Applied before the scale normalisation and the tail selection.

reduction:
  - 'nonzero' (default): mean over the active pairs (hinge > 0), as OpenGait's TripletLoss (AvgNonZeroReducer),
    so the loss does not fade out as most pairs satisfy the margin and its scale is comparable to the triplet.
  - 'mean': mean over all |P_M| |P_NM| pairs of the fold (the formula as written).
"""
import math

import torch
from torch.nn import functional as F

from .base import BaseLoss, gather_and_scale_wrapper
from .open_set import compute_distance


class OpenPairLoss(BaseLoss):
    def __init__(self, num_folds=4, margin=0.2, reduction='nonzero', memory_bank=False, num_classes=None,
                 bank_momentum=0.5, scale_reference='none', scale_detach=False, tail_frac=None,
                 nm_sequences=False, loss_term_weight=1.0):
        super(OpenPairLoss, self).__init__(loss_term_weight)
        if reduction not in ('nonzero', 'mean'):
            raise ValueError(f"reduction must be 'nonzero' or 'mean', got {reduction!r}")
        if memory_bank and not num_classes:
            raise ValueError('memory_bank needs num_classes (number of training identities)')
        if not 0 <= bank_momentum < 1:
            raise ValueError(f'bank_momentum must be in [0, 1), got {bank_momentum}')
        if scale_reference not in ('none', 'genuine'):
            raise ValueError(f"scale_reference must be 'none' or 'genuine', got {scale_reference!r}")
        if tail_frac is not None and not 0 < tail_frac <= 1:
            raise ValueError(f'tail_frac must be in (0, 1], got {tail_frac}')
        self.scale_reference = scale_reference
        self.scale_detach = scale_detach
        self.tail_frac = tail_frac
        self.nm_sequences = nm_sequences
        self.num_folds = num_folds
        self.margin = margin
        self.reduction = reduction
        self.memory_bank = memory_bank
        self.num_classes = num_classes
        self.bank_momentum = bank_momentum
        # Plain tensors (not buffers): the bank is not part of the state dict / checkpoints.
        self.bank = None  # [p, num_classes, c], float32
        self.bank_seen = None  # [num_classes] bool

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

    def bank_min_distance(self, emb, labels):
        """min_j d(x, b_j) over the seen bank templates of the other identities: emb [p, n, c] -> [p, n] (inf if none)."""
        dist = compute_distance(self.bank, emb)  # [p, K, n]; bank is a constant
        own = torch.arange(self.num_classes, device=labels.device).unsqueeze(1) == labels.unsqueeze(0)  # [K, n]
        invalid = own | ~self.bank_seen.unsqueeze(1)
        return dist.masked_fill(invalid.unsqueeze(0), float('inf')).min(1)[0]

    @torch.no_grad()
    def update_bank(self, embeddings, labels):
        """EMA of the per-identity mean embedding of this batch. embeddings [n, c, p]."""
        emb = embeddings.detach().float().permute(2, 0, 1)  # [p, n, c]
        p, _, c = emb.shape
        if self.bank is None or self.bank.device != emb.device:
            self.bank = torch.zeros(p, self.num_classes, c, device=emb.device)
            self.bank_seen = torch.zeros(self.num_classes, dtype=torch.bool, device=emb.device)
        ids, inv = torch.unique(labels, return_inverse=True)
        means = torch.zeros(p, len(ids), c, device=emb.device).index_add_(1, inv, emb)
        means /= torch.bincount(inv, minlength=len(ids)).float().view(1, -1, 1)
        seen = self.bank_seen[ids].view(1, -1, 1)
        m = self.bank_momentum
        # Out of place: this step's distances to the bank are still in the autograd graph (backward comes later).
        bank = self.bank.clone()
        bank[:, ids] = torch.where(seen, m * self.bank[:, ids] + (1 - m) * means, means)
        self.bank = bank
        self.bank_seen = self.bank_seen.clone()
        self.bank_seen[ids] = True

    @gather_and_scale_wrapper
    def forward(self, embeddings, labels):
        # embeddings: [n, c, p], labels: [n]
        unique_labels = torch.unique(labels)
        folds = torch.chunk(unique_labels, self.num_folds)
        if self.memory_bank and labels.max() >= self.num_classes:
            raise ValueError(f'label {int(labels.max())} >= num_classes {self.num_classes}')
        use_bank = self.memory_bank and self.bank is not None and bool(self.bank_seen.any())
        if use_bank:
            # every sample is non-mated in exactly one fold, so compute its bank distance once
            d_bank_all = self.bank_min_distance(embeddings.permute(2, 0, 1).contiguous().float(), labels)  # [p, n]

        hinges, d_gen_all, d_nm_all, d_nm_batch_all, from_bank, scales, from_seq = [], [], [], [], [], [], []
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
            d_nm = compute_distance(gallery, emb_u).min(1)[0]  # [p, n_u]: min over the in-batch templates
            d_nm_batch_all.append(d_nm.detach())
            if self.nm_sequences:
                known_idx = torch.cat(gallery_idx).tolist() + probe_match  # every sequence of the non-fold subjects
                emb_k = embeddings[known_idx].permute(2, 0, 1).contiguous().float()  # [p, n_k, c]
                d_nm_seq = compute_distance(emb_k, emb_u).min(1)[0]  # [p, n_u]
                from_seq.append((d_nm_seq < d_nm).float().detach())
                d_nm = torch.minimum(d_nm, d_nm_seq)  # min over templates U individual sequences
            if use_bank:
                d_nm_bank = d_bank_all[:, probe_nonmatch]  # [p, n_u]
                from_bank.append((d_nm_bank < d_nm).float().detach())
                d_nm = torch.minimum(d_nm, d_nm_bank)  # min over in-batch templates U bank

            d_gen_all.append(d_gen.detach())
            d_nm_all.append(d_nm.detach())
            if self.scale_reference == 'genuine':
                scale = d_gen.mean(1, keepdim=True)  # [p, 1]: mean genuine distance of the fold
                if self.scale_detach:
                    scale = scale.detach()
                scales.append(scale.detach().squeeze(1))
                d_gen, d_nm = d_gen / scale, d_nm / scale
            if self.tail_frac is not None:
                k = max(1, math.ceil(self.tail_frac * d_nm.shape[1]))
                d_nm = d_nm.topk(k, dim=1, largest=False)[0]  # [p, k]: hardest intruders, per part

            hinges.append(F.relu(d_gen.unsqueeze(2) - d_nm.unsqueeze(1) + self.margin).flatten(1))  # [p, n_m * k]

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
            'd_nm_min': torch.cat(d_nm_all, 1).mean(1),  # distance used in the hinge
        })
        if self.nm_sequences:
            self.info['d_nm_min_tmpl'] = torch.cat(d_nm_batch_all, 1).mean(1)  # templates only
            self.info['nm_from_seq'] = torch.cat(from_seq, 1).mean(1)
        if scales:
            self.info['scale'] = torch.stack(scales, 1).mean(1)
        if self.memory_bank:
            self.info['d_nm_min_batch'] = torch.cat(d_nm_batch_all, 1).mean(1)  # in-batch templates only
            self.info['nm_from_bank'] = (torch.cat(from_bank, 1).mean(1) if from_bank
                                         else torch.zeros_like(loss))  # min given by a bank template
            self.update_bank(embeddings, labels)
            self.info['bank_filled'] = self.bank_seen.float().mean().expand_as(loss).clone()
        return loss, self.info
