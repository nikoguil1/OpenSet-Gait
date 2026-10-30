"""Unit tests for opengait/modeling/losses/open_set.py (OpenSetLoss, Su et al. ECCV 2024) on CPU.

Run: python tests/test_open_set_loss.py   (or pytest tests/)
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'opengait'))
from modeling.losses.open_set import OpenSetLoss  # noqa: E402

N_IDS, N_SEQ, C, P = 32, 4, 256, 16
GAIT3D_CFG = dict(num_folds=4, rank=1, score_scale=6.0, rank_scale=0.2, rank_diff_scale=6.0, num_parts=P)


def loss_of(embeddings, labels, **cfg):
    """forward() without the DDP gather wrapper."""
    loss_fn = OpenSetLoss(**{**GAIT3D_CFG, **cfg})
    return OpenSetLoss.forward.__wrapped__(loss_fn, embeddings=embeddings, labels=labels)


def batch(noise, seed=0):
    """TripletSampler-like batch: N_IDS subjects x N_SEQ sequences, embeddings [n, c, p]."""
    g = torch.Generator().manual_seed(seed)
    labels = torch.arange(N_IDS).repeat_interleave(N_SEQ)
    centers = torch.randn(N_IDS, C, P, generator=g)
    return centers[labels] + noise * torch.randn(N_IDS * N_SEQ, C, P, generator=g), labels


def test_shapes_and_finite_gradients():
    emb, labels = batch(noise=1.0)
    emb.requires_grad_(True)
    loss, info = loss_of(emb, labels)
    assert loss.shape == (P,) and torch.isfinite(loss).all()
    assert info['dirs'].shape == (P,) and info['nm_scores'].shape == (P,)
    loss.mean().backward()
    assert torch.isfinite(emb.grad).all() and emb.grad.abs().sum() > 0


def test_separated_identities_lower_loss():
    # mated probes close to their template, non-mated far from every template
    sep, labels = batch(noise=0.05)
    rnd = torch.randn_like(sep)
    loss_sep, info_sep = loss_of(sep, labels)
    loss_rnd, info_rnd = loss_of(rnd, labels)
    assert loss_sep.mean() < loss_rnd.mean()
    assert info_sep['dirs'].mean() > info_rnd['dirs'].mean()


def test_rtm_weight():
    emb, labels = batch(noise=1.0)
    l0, info = loss_of(emb, labels, rtm_weight=0.0)
    l4, _ = loss_of(emb, labels, rtm_weight=4.0)
    assert torch.allclose(l0, 1 - info['dirs'])
    assert torch.allclose(l4 - l0, 4 * info['nm_scores'])


def test_zero_distance_gradient_is_finite():
    # identical sequences give d = 0 between template and probe; sqrt'(0) must not produce NaN
    labels = torch.arange(N_IDS).repeat_interleave(N_SEQ)
    emb = torch.randn(N_IDS, C, P)[labels].clone().requires_grad_(True)
    loss, _ = loss_of(emb, labels)
    loss.mean().backward()
    assert torch.isfinite(emb.grad).all()


if __name__ == '__main__':
    tests = [v for k, v in dict(globals()).items() if k.startswith('test_')]
    for t in tests:
        t()
        print(f'OK  {t.__name__}')
    print(f'{len(tests)} tests OK')
