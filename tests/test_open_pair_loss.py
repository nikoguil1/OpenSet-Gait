"""Unit tests for opengait/modeling/losses/open_pair.py (OpenPairLoss) on CPU.

Run: python tests/test_open_pair_loss.py   (or pytest tests/)
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'opengait'))
from modeling.losses.open_pair import OpenPairLoss  # noqa: E402
from modeling.losses.open_set import compute_distance  # noqa: E402

N_IDS, N_SEQ, C, P = 32, 4, 64, 15


def loss_of(embeddings, labels, **cfg):
    loss_fn = OpenPairLoss(**cfg)
    return OpenPairLoss.forward.__wrapped__(loss_fn, embeddings=embeddings, labels=labels)


def batch(noise, seed=0, perm=False):
    g = torch.Generator().manual_seed(seed)
    labels = torch.arange(N_IDS).repeat_interleave(N_SEQ) * 7 + 3  # non-contiguous ids
    centers = torch.randn(N_IDS, C, P, generator=g)
    emb = centers.repeat_interleave(N_SEQ, 0) + noise * torch.randn(N_IDS * N_SEQ, C, P, generator=g)
    if perm:
        idx = torch.randperm(len(labels), generator=g)
        emb, labels = emb[idx], labels[idx]
    return emb, labels


def reference(emb, labels, num_folds=4, margin=0.2):
    """Literal double loop over (x_m, x_u) pairs, 'mean' reduction pooled over folds."""
    uniq = torch.unique(labels)
    terms = []
    for fold in torch.chunk(uniq, num_folds):
        G, gl, pm, pu = OpenPairLoss.split_fold(labels, uniq, fold)
        gal = torch.stack([emb[i].mean(0) for i in G]).permute(2, 0, 1)
        gl = torch.stack(gl)
        dm = compute_distance(gal, emb[pm].permute(2, 0, 1))
        du = compute_distance(gal, emb[pu].permute(2, 0, 1))
        for a, xm in enumerate(pm):
            gi = (gl == labels[xm]).nonzero().item()
            for b in range(len(pu)):
                terms.append(torch.relu(dm[:, gi, a] - du[:, :, b].min(1)[0] + margin))
    return torch.stack(terms, 1).mean(1)


def test_matches_formula():
    emb, labels = batch(noise=1.0, perm=True)
    emb, labels = emb[:64], labels[:64]  # unbalanced subjects, smaller loop
    loss, _ = loss_of(emb, labels, reduction='mean')
    assert torch.allclose(loss, reference(emb, labels), atol=1e-5)


def test_partition_no_leakage():
    _, labels = batch(noise=1.0, perm=True)
    uniq = torch.unique(labels)
    for fold in torch.chunk(uniq, 4):
        G, gl, pm, pu = OpenPairLoss.split_fold(labels, uniq, fold)
        g_idx = set(torch.cat(G).tolist())
        assert not g_idx & set(pm) and not g_idx & set(pu) and not set(pm) & set(pu)
        assert set(labels[pu].tolist()) == set(fold.tolist())
        assert not set(torch.stack(gl).tolist()) & set(fold.tolist())
        assert len(g_idx) == len(pm) == 2 * len(gl)  # 4 seqs: 2 template + 2 mated


def test_shapes_gradients_and_separation():
    emb, labels = batch(noise=1.0)
    emb.requires_grad_(True)
    loss, info = loss_of(emb, labels)
    assert loss.shape == (P,) and torch.isfinite(loss).all()
    loss.mean().backward()
    assert torch.isfinite(emb.grad).all() and emb.grad.abs().sum() > 0
    sep, _ = batch(noise=0.01)
    l_sep, info_sep = loss_of(sep, labels)
    assert l_sep.abs().max() == 0 and info_sep['active_frac'].max() == 0
    assert info['active_frac'].min() > 0


def test_nonzero_reduction():
    emb, labels = batch(noise=1.0)
    l_mean, info = loss_of(emb, labels, reduction='mean')
    l_nz, _ = loss_of(emb, labels, reduction='nonzero')
    assert torch.allclose(l_nz * info['active_frac'], l_mean, atol=1e-5)


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print(f'{name}: OK')
