"""Unit tests for opengait/modeling/losses/open_pair.py (OpenPairLoss) on CPU.

Run: python tests/test_open_pair_loss.py   (or pytest tests/)
"""
import math
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


# --- scale_reference='genuine' (idea 1) and tail_frac (idea 2) ---

def reference_v2(emb, labels, num_folds=4, margin=0.2, scale=False, tail_frac=None):
    """Literal loops: per fold and part, divide by the mean genuine distance and keep the ceil(beta |P_NM|) closest
    intruders; 'mean' reduction pooled over folds."""
    uniq = torch.unique(labels)
    terms = []
    for fold in torch.chunk(uniq, num_folds):
        G, gl, pm, pu = OpenPairLoss.split_fold(labels, uniq, fold)
        gal = torch.stack([emb[i].mean(0) for i in G]).permute(2, 0, 1)
        gl = torch.stack(gl)
        dm = compute_distance(gal, emb[pm].permute(2, 0, 1))
        du = compute_distance(gal, emb[pu].permute(2, 0, 1)).min(1)[0]  # [p, n_u]
        dg = torch.stack([dm[:, (gl == labels[xm]).nonzero().item(), a] for a, xm in enumerate(pm)], 1)  # [p, n_m]
        for part in range(dg.shape[0]):
            g, u = dg[part], du[part]
            if scale:
                g, u = g / g.mean(), u / g.mean()
            if tail_frac is not None:
                k = max(1, math.ceil(tail_frac * len(u)))
                u = torch.sort(u)[0][:k]
            for a in range(len(g)):
                for b in range(len(u)):
                    terms.append((part, torch.relu(g[a] - u[b] + margin)))
    p = emb.shape[2]
    return torch.stack([torch.stack([t for q, t in terms if q == part]).mean() for part in range(p)])


def test_scale_and_tail_match_formula():
    emb, labels = batch(noise=1.0, perm=True, seed=3)
    emb, labels = emb[:64], labels[:64]
    for cfg in [dict(scale=True), dict(tail_frac=0.25), dict(scale=True, tail_frac=0.25)]:
        loss, _ = loss_of(emb, labels, reduction='mean', scale_reference='genuine' if cfg.get('scale') else 'none',
                          tail_frac=cfg.get('tail_frac'))
        assert torch.allclose(loss, reference_v2(emb, labels, **cfg), atol=1e-5), cfg


def test_tail_frac_one_equals_plain():
    emb, labels = batch(noise=1.0, perm=True)
    plain, _ = loss_of(emb, labels)
    full, _ = loss_of(emb, labels, tail_frac=1.0)
    assert torch.allclose(plain, full)


def test_tail_is_harder():
    emb, labels = batch(noise=1.0)
    l_all, i_all = loss_of(emb, labels, reduction='mean')
    l_tail, i_tail = loss_of(emb, labels, reduction='mean', tail_frac=0.25)
    assert (l_tail > l_all).all() and (i_tail['active_frac'] > i_all['active_frac']).all()


def test_genuine_scale_invariance_and_no_radial_gradient():
    emb, labels = batch(noise=1.0, seed=5)
    cfg = dict(scale_reference='genuine', tail_frac=0.25)
    l1, info = loss_of(emb, labels, **cfg)
    l3, _ = loss_of(3.0 * emb, labels, **cfg)
    assert torch.allclose(l1, l3, atol=1e-5)
    assert torch.allclose(info['scale'], info['d_gen'], rtol=1e-4)  # same folds, mean of means
    x = emb.clone().requires_grad_(True)
    loss, _ = loss_of(x, labels, **cfg)
    loss.sum().backward()
    radial = (x.grad * x).sum() / (x.grad.norm() * x.norm())
    assert radial.abs() < 1e-4 and x.grad.abs().sum() > 0
    # with the detached scale the loss is no longer scale invariant
    ld, _ = loss_of(emb, labels, scale_detach=True, **cfg)
    assert torch.allclose(ld, l1, atol=1e-6)  # same value...
    y = emb.clone().requires_grad_(True)
    loss_d, _ = loss_of(y, labels, scale_detach=True, **cfg)
    loss_d.sum().backward()
    assert ((y.grad * y).sum() / (y.grad.norm() * y.norm())).abs() > 1e-3  # ...but a radial gradient


def test_rejects_bad_v2_args():
    for cfg in [dict(scale_reference='all'), dict(tail_frac=0.0), dict(tail_frac=1.5)]:
        try:
            OpenPairLoss(**cfg)
        except ValueError:
            continue
        raise AssertionError(f'{cfg} must raise')


# --- nm_sequences (idea 3a) ---

def reference_seq(emb, labels, num_folds=4, margin=0.2):
    """Literal loops: d_nm = min over the fold templates and every sequence of the non-fold subjects; 'mean'."""
    uniq = torch.unique(labels)
    terms = []
    for fold in torch.chunk(uniq, num_folds):
        G, gl, pm, pu = OpenPairLoss.split_fold(labels, uniq, fold)
        gal = torch.stack([emb[i].mean(0) for i in G]).permute(2, 0, 1)
        gl = torch.stack(gl)
        known = [i for i in range(len(labels)) if labels[i].item() not in fold.tolist()]
        cand = torch.cat([gal, emb[known].permute(2, 0, 1)], 1)  # [p, n_g + n_k, c]
        dm = compute_distance(gal, emb[pm].permute(2, 0, 1))
        du = compute_distance(cand, emb[pu].permute(2, 0, 1))
        for a, xm in enumerate(pm):
            gi = (gl == labels[xm]).nonzero().item()
            for b in range(len(pu)):
                terms.append(torch.relu(dm[:, gi, a] - du[:, :, b].min(1)[0] + margin))
    return torch.stack(terms, 1).mean(1)


def test_nm_sequences_matches_formula():
    emb, labels = batch(noise=1.0, perm=True, seed=7)
    emb, labels = emb[:64], labels[:64]
    loss, info = loss_of(emb, labels, reduction='mean', nm_sequences=True)
    assert torch.allclose(loss, reference_seq(emb, labels), atol=1e-5)
    assert (info['d_nm_min'] <= info['d_nm_min_tmpl'] + 1e-6).all()
    assert (info['nm_from_seq'] > 0).all()


def test_nm_sequences_off_equals_plain_and_grads():
    emb, labels = batch(noise=1.0, perm=True)
    plain, _ = loss_of(emb, labels)
    off, _ = loss_of(emb, labels, nm_sequences=False)
    assert torch.allclose(plain, off)
    x = emb.clone().requires_grad_(True)
    loss, info = loss_of(x, labels, nm_sequences=True, scale_reference='genuine', tail_frac=0.25)
    loss.sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    l3, _ = loss_of(3.0 * emb, labels, nm_sequences=True, scale_reference='genuine', tail_frac=0.25)
    assert torch.allclose(loss.detach(), l3, atol=1e-5)  # still scale invariant


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print(f'{name}: OK')
