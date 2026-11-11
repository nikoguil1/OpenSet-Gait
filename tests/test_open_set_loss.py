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


def test_detection_only():
    # identification=False: DIR = mean sigma_alpha(s_match - theta), i.e. the IDL without S_id (S_id <= 0.525)
    emb, labels = batch(noise=1.0)
    _, info_id = loss_of(emb, labels, rtm_weight=0.0)
    l1, info_det = loss_of(emb, labels, rtm_weight=0.0, identification=False)
    assert (info_det['dirs'] > info_id['dirs']).all()
    assert torch.allclose(l1, 1 - info_det['dirs'])
    l3, _ = loss_of(emb, labels, rtm_weight=4.0, identification=False, det_weight=3.0)
    assert torch.allclose(l3, 3 * (1 - info_det['dirs']) + 4 * info_det['nm_scores'])
    # hand-computed detection rate of part 0: 4 folds of 8 subjects; template = mean of the first 2 sequences
    alpha = GAIT3D_CFG['score_scale']
    x = emb[:, :, 0]
    dets = []
    for f in range(4):
        known = [s for s in range(N_IDS) if not 8 * f <= s < 8 * (f + 1)]
        gal = torch.stack([x[labels == s][:2].mean(0) for s in known])
        mated = torch.cat([x[labels == s][2:] for s in known])
        s_match = 1 / (1 + (mated - gal.repeat_interleave(2, 0)).norm(dim=1))
        s_nm = 1 / (1 + torch.cdist(gal, x[(labels >= 8 * f) & (labels < 8 * (f + 1))]))
        dets.append(torch.sigmoid(alpha * (s_match[None] - s_nm.reshape(-1, 1))).mean(1))  # per theta
    assert torch.allclose(torch.cat(dets).mean(), info_det['dirs'][0], atol=1e-5)
    emb.requires_grad_(True)
    loss, _ = loss_of(emb, labels, identification=False, det_weight=3.0)
    loss.mean().backward()
    assert torch.isfinite(emb.grad).all() and emb.grad.abs().sum() > 0


def test_logged_identification_and_detection():
    # info['Id'] = mean S_id over mated probes, info['Det'] = mean S_det over (theta, probe), as in dirs
    emb, labels = batch(noise=1.0)
    loss, info = loss_of(emb, labels)
    assert info['Id'].shape == (P,) and info['Det'].shape == (P,)
    assert (info['Id'] > 0).all() and (info['Id'] <= torch.sigmoid(torch.tensor(0.2 * 0.5)) + 1e-6).all()
    assert (info['Det'] > 0).all() and (info['Det'] < 1).all()
    # without identification S_id = 1, so Det equals dirs
    _, info_det = loss_of(emb, labels, identification=False)
    assert torch.allclose(info_det['Id'], torch.ones(P))
    assert torch.allclose(info_det['Det'], info_det['dirs'])
    # logging does not change the loss
    l_ref, _ = loss_of(emb, labels, rtm_weight=0.0)
    assert torch.allclose(l_ref, 1 - info['dirs'])


def test_scale_invariant_scores():
    # scale_invariant: d' = d / mean(d) per part and fold, so loss and logged factors do not change with the
    # embedding scale; the default (Su) scores 1/(1+d) do
    emb, labels = batch(noise=1.0)
    kw = dict(scale_invariant=True)
    l1, i1 = loss_of(emb, labels, **kw)
    l5, i5 = loss_of(5 * emb, labels, **kw)
    assert torch.allclose(l1, l5, atol=1e-5)
    for k in ('dirs', 'Id', 'Det'):
        assert torch.allclose(i1[k], i5[k], atol=1e-5)
    assert not torch.allclose(loss_of(emb, labels)[0], loss_of(5 * emb, labels)[0])
    # mean distance -> 1, so the mean scores lie around 1/2 instead of 1/(1+d)
    _, i0 = loss_of(emb, labels)
    assert (i1['dirs'] > i0['dirs']).all()
    emb.requires_grad_(True)
    loss, _ = loss_of(emb, labels, **kw, rank_self_term=False)
    loss.mean().backward()
    assert torch.isfinite(emb.grad).all() and emb.grad.abs().sum() > 0


def test_genuine_scale_reference():
    # scale_reference='genuine': d' = d / mean mated distance, so genuine pairs sit at d' ~ 1 and intruders above;
    # with scale_detach=False the loss is exactly scale invariant, so its gradient is orthogonal to the embeddings
    emb, labels = batch(noise=1.0)
    calib = dict(scale_invariant=True, rank_self_term=False, score_scale=20.0, rank_scale=1.0,
                 rank_diff_scale=30.0)
    kw = dict(calib, scale_reference='genuine', scale_detach=False)
    l1, i1 = loss_of(emb, labels, **kw)
    l5, i5 = loss_of(5 * emb, labels, **kw)
    assert torch.allclose(l1, l5, atol=1e-5)
    for k in ('dirs', 'Id', 'Det', 'nm_scores'):
        assert torch.allclose(i1[k], i5[k], atol=1e-5)
    # intruders above the genuine mean: lower RTM scores than with the all-pairs mean (E4.5)
    _, i_all = loss_of(emb, labels, **calib)
    assert (i1['nm_scores'] < i_all['nm_scores']).all()
    assert (i1['nm_scores'] < 0.5).all()

    def radial_cosine(**cfg):
        x = emb.clone().requires_grad_(True)
        loss_of(x, labels, **cfg)[0].mean().backward()
        return ((x.grad * x).sum() / (x.grad.norm() * x.norm())).abs().item()

    assert radial_cosine(**kw) < 1e-4  # no pressure to inflate (or shrink) the distances
    assert radial_cosine(**calib) > 1e-2  # detached mean: radial component (E4.5 inflated mean_dist)
    x = emb.clone().requires_grad_(True)
    loss_of(x, labels, **kw)[0].mean().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    try:
        OpenSetLoss(scale_reference='intruder')
        raise AssertionError('invalid scale_reference accepted')
    except ValueError:
        pass


def test_rank_without_self_term():
    # the own template contributes sigma(0) = 0.5 to the soft rank; rank_self_term=False removes it, so
    # S_id = sigma(beta * (1 - (soft_rank - 0.5))): upper bound sigma(beta) instead of sigma(beta / 2)
    emb, labels = batch(noise=0.05)  # well separated + sharp gamma: soft rank ~ own term only
    beta = 1.0
    _, i_self = loss_of(emb, labels, rank_scale=beta, rank_diff_scale=50.0)
    _, i_nos = loss_of(emb, labels, rank_scale=beta, rank_diff_scale=50.0, rank_self_term=False)
    assert torch.allclose(i_self['Id'], torch.sigmoid(torch.tensor(beta * 0.5)).expand(P), atol=1e-3)
    assert torch.allclose(i_nos['Id'], torch.sigmoid(torch.tensor(beta)).expand(P), atol=1e-3)
    assert torch.allclose(i_self['Det'], i_nos['Det'])


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
