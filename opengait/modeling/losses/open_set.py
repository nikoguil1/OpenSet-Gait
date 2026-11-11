"""Open-set loss of Su et al., "Open-Set Biometrics: Beyond Good Closed-Set Models", ECCV 2024.

Ported from https://github.com/prevso1088/open-set-biometrics (commit 76ea429,
opengait/modeling/losses/open_set.py; the repository has no license file).
Each minibatch simulates open-set evaluation: the subjects are split into folds, each
fold in turn acts as non-mated probes, and the remaining subjects are split into a
mean gallery template (first half of their sequences) and mated probes (second half).

  - Identification-detection loss (IDL, eq. 8-11): soft DIR = sigmoid(beta * (r - soft_rank))
    * sigmoid(alpha * (s_match - theta)), with every non-mated score used as threshold theta.
  - Relative threshold minimization (RTM, eq. 12): softmax-weighted mean of non-mated scores.
  - loss = 1 - DIR + lambda * RTM, per part.

Scores are s = 1 / (1 + d) on Euclidean distances (eq. 6).
Changes from the original: lambda (`rtm_weight`, hard-coded 4) and loss_term_weight are
arguments; the distance has an epsilon inside the sqrt (finite gradient at d = 0).

Optional recalibration (off by default, CU [EXP] IDL calibrada):
  - scale_invariant: scores use d' = d / mean(d), the mean taken (detached) per part and fold over all
    gallery-probe distances (mated and non-mated). Scores are then invariant to the embedding scale,
    so neither the IDL nor the RTM can be lowered by inflating the distances.
    The RTM softmax weights become softmax(-rtm_softmax_scale * d') (the original softmax(-d) is
    equivalent to rtm_softmax_scale = mean distance, ~5-12 for SwinGait-3D on Gait3D).
  - scale_reference='genuine' (with scale_invariant): divide by the mean mated (genuine) distance instead
    of the mean over all pairs. Genuine pairs sit at d' ~ 1 and intruders above, so the RTM keeps a margin
    to push (with the all-pairs mean, dominated by impostor pairs, intruders stay at d' ~ 1 and the RTM
    is flat; E4.5). scale_detach=False lets the gradient flow through the mean: the loss is then exactly
    invariant to the embedding scale and its gradient has no radial component, so it cannot inflate the
    distances (with the detached mean, mean_dist kept growing in E4.5).
  - rank_self_term=False: the soft rank drops the own-template term sigma(0) = 0.5, so a probe with no
    impostor above it has soft rank ~0 and S_id = sigma(beta * r) (instead of sigma(beta * (r - 0.5))).
"""
import torch
from torch import nn
from torch.nn import functional as F

from .base import BaseLoss, gather_and_scale_wrapper


class BatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(BatchNorm1d, self).__init__()
        self.bn = nn.BatchNorm1d(num_features, eps, momentum, affine)

    def forward(self, x):
        """
            x: [p ...]
        """
        p, *rest = x.shape
        x = x.view(p, -1).permute(1, 0).contiguous()  # [_, p]
        x = self.bn(x)
        x = x.permute(1, 0).contiguous().view(p, *rest)
        return x


class NormedSigmoid(nn.Module):
    def __init__(self, num_parts=16, scale=1., norm=False):
        super(NormedSigmoid, self).__init__()
        self.norm = BatchNorm1d(num_parts, affine=False) if norm else nn.Identity()
        self.scale = scale

    def forward(self, x):
        """
            x: [p, ....]
        """
        return torch.sigmoid(self.scale * self.norm(x))


def compute_distance(x, y, eps=1e-12):
    """
        x: [p, n_x, c]
        y: [p, n_y, c]
    """
    x2 = torch.sum(x ** 2, -1).unsqueeze(2)  # [p, n_x, 1]
    y2 = torch.sum(y ** 2, -1).unsqueeze(1)  # [p, 1, n_y]
    inner = x.matmul(y.transpose(1, 2))  # [p, n_x, n_y]
    dist = x2 + y2 - 2 * inner
    return torch.sqrt(F.relu(dist) + eps)  # [p, n_x, n_y]


def find_matches(scores, labels_g, labels_p):
    """
        scores: [p, n_g, n_p]
        labels_g: [n_g]
        labels_p: [n_p]
    """
    # mask indicating matching gallery-probe pairs
    match_mask = labels_g.unsqueeze(1) == labels_p.unsqueeze(0)

    # use the match mask to extract the scores
    return scores[:, None, match_mask]  # Shape: [p, 1, n_matches]


class OpenSetLoss(BaseLoss):
    def __init__(self, num_folds=4, rank=1, score_scale=9.0, rank_scale=1.5, rank_diff_scale=9.0,
                 num_parts=16, rtm_weight=4.0, identification=True, det_weight=1.0,
                 scale_invariant=False, rank_self_term=True, rtm_softmax_scale=10.0,
                 scale_reference='all', scale_detach=True, loss_term_weight=1.0):
        super(OpenSetLoss, self).__init__(loss_term_weight)
        self.num_folds = num_folds
        self.rank = rank
        self.num_parts = num_parts
        self.rtm_weight = rtm_weight
        # identification=False drops S_id from the IDL: only the detection term sigma_alpha(s_match - theta)
        # remains, weighted by det_weight (CU [EXP] OpenSetLoss sin identificación)
        self.identification = identification
        self.det_weight = det_weight
        self.scale_invariant = scale_invariant
        self.rank_self_term = rank_self_term
        self.rtm_softmax_scale = rtm_softmax_scale
        if scale_reference not in ('all', 'genuine'):
            raise ValueError(f"scale_reference must be 'all' or 'genuine', got {scale_reference!r}")
        self.scale_reference = scale_reference
        self.scale_detach = scale_detach

        self.rank_sigmoid = NormedSigmoid(num_parts=num_parts, scale=rank_scale)  # beta
        self.thres_sigmoid = NormedSigmoid(num_parts=num_parts, scale=score_scale)  # alpha
        self.rank_diff_sigmoid = NormedSigmoid(num_parts=num_parts, scale=rank_diff_scale)  # gamma

    @gather_and_scale_wrapper
    def forward(self, embeddings, labels):
        # embeddings: [n, c, p], label: [n]

        # divide gallery into folds
        unique_labels = torch.unique(labels)
        folds = torch.chunk(unique_labels, self.num_folds)

        dirs = []
        softmax_scores_nm = []
        ids, dets = [], []  # logging only: identification S_id and detection S_det factors of the soft DIR
        for fold in folds:
            embeddings_gallery = []
            labels_gallery = []
            probe_match_set = []
            probe_nonmatch_set = []

            for subject in unique_labels:
                # get embeddings and labels for the current subject
                subject_indices = torch.nonzero(labels == subject, as_tuple=False).view(-1)

                if subject in fold:
                    # add everything to probe
                    probe_nonmatch_set.extend(subject_indices.tolist())
                else:
                    # divide the embeddings into gallery and probe sets
                    num_sequences = len(subject_indices)
                    split_index = max(1, int(num_sequences / 2 + .5))
                    embeddings_gallery.append(embeddings[subject_indices[:split_index]].mean(0))
                    labels_gallery.append(subject)
                    probe_match_set.extend(subject_indices[split_index:].tolist())

            # compute gallery-probe distances
            embeddings_gallery = torch.stack(
                embeddings_gallery).permute(2, 0, 1).contiguous().float()  # [n, c, p] -> [p, n, c]
            embeddings_probe = embeddings[probe_match_set, :].permute(
                2, 0, 1).contiguous().float()  # [n, c, p] -> [p, n, c]
            dist_fold = compute_distance(embeddings_gallery, embeddings_probe)  # [p, n_g, n_p]

            embeddings_nm = embeddings[probe_nonmatch_set, :].permute(
                2, 0, 1).contiguous().float()  # [n, c, p] -> [p, n, c]
            dist_nm = compute_distance(embeddings_gallery, embeddings_nm)  # [p, n_g, n_p]

            labels_gallery = torch.as_tensor(labels_gallery, device=labels.device)
            labels_probe = torch.as_tensor(labels[probe_match_set], device=labels.device)

            # convert from distances to scores
            if self.scale_invariant:
                if self.scale_reference == 'genuine':
                    scale = find_matches(dist_fold, labels_gallery, labels_probe).mean(-1, keepdim=True)  # [p, 1, 1]
                else:
                    scale = torch.cat([dist_fold.flatten(1), dist_nm.flatten(1)], 1).mean(1).view(-1, 1, 1)
                if self.scale_detach:
                    scale = scale.detach()
                dist_fold, dist_nm = dist_fold / scale, dist_nm / scale
                rtm_logits = -self.rtm_softmax_scale * dist_nm
            else:
                rtm_logits = -dist_nm
            scores_fold = 1 / (1 + dist_fold)
            scores_nm = 1 / (1 + dist_nm)

            # whether each probe's rank is less than r
            scores_match = find_matches(scores_fold, labels_gallery, labels_probe)  # [p, 1, n_matches]
            if self.identification:
                ranks = self.soft_rank(scores_fold, scores_match)
                if not self.rank_self_term:
                    ranks = ranks - 0.5  # own template: sigmoid(gamma * 0) = 0.5 exactly
                ranks = self.rank_sigmoid(self.rank - ranks)
            else:
                ranks = torch.ones_like(scores_match[:, 0])  # [p, n_matches]: detection only

            softmax_scores_nm.append(torch.sum(torch.softmax(rtm_logits, dim=1) * scores_nm, dim=1))  # [p, n_p]

            # use the set of scores as thresholds for FNIR
            thetas = scores_nm.view(scores_nm.shape[0], -1, 1)  # [p, num_thetas, 1]
            thresh = self.thres_sigmoid(scores_match - thetas)
            dirs.append(torch.mean(ranks.unsqueeze(1) * thresh, dim=-1))  # dir: [p, num_thetas, n_p]
            ids.append(ranks.detach())  # [p, n_matches]
            dets.append(thresh.detach().mean(-1))  # [p, num_thetas], averaged like dirs

        dirs = torch.cat(dirs, dim=1).contiguous().view(self.num_parts, -1)
        softmax_scores_nm = torch.cat(softmax_scores_nm, dim=1).contiguous().view(self.num_parts, -1)
        losses = self.det_weight * (1 - dirs.mean(1)) + self.rtm_weight * softmax_scores_nm.mean(1)

        self.info.update({
            'loss': losses.detach().clone(),
            'dirs': dirs.mean(1),
            'nm_scores': softmax_scores_nm.mean(1),
            'Id': torch.cat(ids, dim=1).mean(1),
            'Det': torch.cat(dets, dim=1).mean(1),
        })

        for name, child in self.named_modules():
            if isinstance(child, BatchNorm1d):
                self.info.update({
                    f'{name}_mean': child.bn.running_mean,
                    f'{name}_std': child.bn.running_var.sqrt(),
                })

        return losses, self.info

    def soft_rank(self, scores, scores_match):
        """
            scores: [p, n_g, n_p]
            scores_match: [p, 1, n_p]
        """
        scores_diff = scores - scores_match  # Shape: [p, n_g, n_p]

        # compute the soft rank
        return torch.sum(self.rank_diff_sigmoid(scores_diff), dim=1)  # Shape: [p, n_p]
