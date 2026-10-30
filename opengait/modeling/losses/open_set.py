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
                 num_parts=16, rtm_weight=4.0, loss_term_weight=1.0):
        super(OpenSetLoss, self).__init__(loss_term_weight)
        self.num_folds = num_folds
        self.rank = rank
        self.num_parts = num_parts
        self.rtm_weight = rtm_weight

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

            # convert from distances to scores
            scores_fold = 1 / (1 + dist_fold)
            scores_nm = 1 / (1 + dist_nm)

            # whether each probe's rank is less than r
            labels_gallery = torch.as_tensor(labels_gallery, device=labels.device)
            labels_probe = torch.as_tensor(labels[probe_match_set], device=labels.device)
            scores_match = find_matches(scores_fold, labels_gallery, labels_probe)  # [p, 1, n_matches]
            ranks = self.soft_rank(scores_fold, scores_match)
            ranks = self.rank_sigmoid(self.rank - ranks)

            softmax_scores_nm.append(torch.sum(torch.softmax(-dist_nm, dim=1) * scores_nm, dim=1))  # [p, n_p]

            # use the set of scores as thresholds for FNIR
            thetas = scores_nm.view(scores_nm.shape[0], -1, 1)  # [p, num_thetas, 1]
            thresh = self.thres_sigmoid(scores_match - thetas)
            dirs.append(torch.mean(ranks.unsqueeze(1) * thresh, dim=-1))  # dir: [p, num_thetas, n_p]

        dirs = torch.cat(dirs, dim=1).contiguous().view(self.num_parts, -1)
        softmax_scores_nm = torch.cat(softmax_scores_nm, dim=1).contiguous().view(self.num_parts, -1)
        losses = 1 - dirs.mean(1) + self.rtm_weight * softmax_scores_nm.mean(1)

        self.info.update({
            'loss': losses.detach().clone(),
            'dirs': dirs.mean(1),
            'nm_scores': softmax_scores_nm.mean(1),
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
