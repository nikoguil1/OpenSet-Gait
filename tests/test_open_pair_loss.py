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


K = (N_IDS - 1) * 7 + 3 + 1  # labels are class indices < num_classes


def call(loss_fn, emb, labels):
    return OpenPairLoss.forward.__wrapped__(loss_fn, embeddings=emb, labels=labels)


def test_bank_empty_equals_plain_loss():
    emb, labels = batch(noise=1.0, perm=True)
    plain, _ = loss_of(emb, labels)
    banked, info = call(OpenPairLoss(memory_bank=True, num_classes=K), emb, labels)
    assert torch.allclose(plain, banked) and info['nm_from_bank'].abs().max() == 0
    assert torch.allclose(info['d_nm_min'], info['d_nm_min_batch'])


def test_bank_update_mean_and_ema():
    emb, labels = batch(noise=1.0, perm=True)
    lf = OpenPairLoss(memory_bank=True, num_classes=K, bank_momentum=0.5)
    call(lf, emb, labels)
    j = int(labels[0])
    mean_j = emb[labels == j].mean(0).permute(1, 0)  # [p, c]
    assert torch.allclose(lf.bank[:, j], mean_j, atol=1e-6)
    assert lf.bank_seen.sum() == N_IDS and not lf.bank.requires_grad
    emb2, labels2 = batch(noise=1.0, seed=1, perm=True)
    call(lf, emb2, labels2)
    assert torch.allclose(lf.bank[:, j], 0.5 * mean_j + 0.5 * emb2[labels2 == j].mean(0).permute(1, 0), atol=1e-6)


def test_bank_matches_formula():
    emb, labels = batch(noise=1.0, perm=True)
    emb, labels = emb[:64], labels[:64]
    lf = OpenPairLoss(memory_bank=True, num_classes=K, reduction='mean')
    bank_src, bank_lbl = batch(noise=1.0, seed=3)
    call(lf, bank_src[:48], bank_lbl[:48])  # bank with 12 identities, some of them in the batch
    bank, seen = lf.bank.clone(), lf.bank_seen.clone()
    loss, _ = call(lf, emb, labels)
    uniq, terms = torch.unique(labels), []
    for fold in torch.chunk(uniq, 4):
        G, gl, pm, pu = OpenPairLoss.split_fold(labels, uniq, fold)
        gal = torch.stack([emb[i].mean(0) for i in G]).permute(2, 0, 1)
        gl = torch.stack(gl)
        dm = compute_distance(gal, emb[pm].permute(2, 0, 1))
        du = compute_distance(gal, emb[pu].permute(2, 0, 1))
        db = compute_distance(bank, emb[pu].permute(2, 0, 1))  # [p, K, n_u]
        for a, xm in enumerate(pm):
            gi = (gl == labels[xm]).nonzero().item()
            for b, xu in enumerate(pu):
                others = [k for k in range(K) if seen[k] and k != int(labels[xu])]
                d_nm = torch.minimum(du[:, :, b].min(1)[0], db[:, others, b].min(1)[0])
                terms.append(torch.relu(dm[:, gi, a] - d_nm + 0.2))
    assert torch.allclose(loss, torch.stack(terms, 1).mean(1), atol=1e-5)


def test_bank_excludes_own_identity_and_has_no_gradient():
    emb, labels = batch(noise=1.0, perm=True)
    lf = OpenPairLoss(memory_bank=True, num_classes=K)
    call(lf, emb, labels)  # bank = exact means of these identities
    emb.requires_grad_(True)
    loss, info = call(lf, emb, labels)
    # own templates are at distance ~noise; if they were not excluded d_nm would collapse towards them
    d_own = (emb.detach() - lf.bank.permute(1, 2, 0)[labels]).norm(dim=1).mean()
    assert info['d_nm_min'].mean() > 1.2 * d_own
    loss.mean().backward()
    assert torch.isfinite(emb.grad).all() and emb.grad.abs().sum() > 0 and lf.bank.grad is None
    assert info['bank_filled'].max() == N_IDS / K


def test_bank_rejects_labels_out_of_range():
    emb, labels = batch(noise=1.0)
    try:
        call(OpenPairLoss(memory_bank=True, num_classes=10), emb, labels)
    except ValueError:
        return
    raise AssertionError('labels >= num_classes must raise')


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print(f'{name}: OK')
