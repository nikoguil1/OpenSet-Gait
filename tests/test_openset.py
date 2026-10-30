"""Unit tests for opengait/evaluation/openset.py on synthetic distances.

Run: python tests/test_openset.py   (or pytest tests/)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'opengait'))
from evaluation.openset import (legacy_split, localscore_dist, make_splits, run_protocol,  # noqa: E402
                                run_su2024, split_metrics, subject_templates, SU2024_CFG, load_su2024_splits)


def euclidean(x, y):
    return np.sqrt(np.maximum((x ** 2).sum(1)[:, None] + (y ** 2).sum(1)[None] - 2 * x @ y.T, 0))


def dist_fn(x, y, metric):
    """cuda_dist convention on numpy: euclidean, or 1 - cosine."""
    if metric == 'cos':
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        return 1 - x @ y.T
    return euclidean(x, y)


def synthetic(n_ids=40, n_gal=3, noise=0.05, seed=0):
    """One cluster per identity; 1 probe and n_gal gallery samples each."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_ids, 32)) * 3
    probe = centers + rng.normal(scale=noise, size=centers.shape)
    gallery = np.repeat(centers, n_gal, 0) + rng.normal(scale=noise, size=(n_ids * n_gal, 32))
    lbls = np.array([f'{i:03d}' for i in range(n_ids)])
    return euclidean(probe, gallery), lbls, np.repeat(lbls, n_gal)


def test_perfect_embeddings():
    dist, p, g = synthetic(noise=0.01)
    m, _ = split_metrics(dist, p, g, unknown_ids=p[:10])
    for k in ('rank1', 'dir@0.01fpir', 'dir@0.1fpir', 'oscr', 'auroc', 'tar@0.01far'):
        assert abs(m[k] - 100) < 1e-6, (k, m[k])
    assert m['eer'] < 1e-6 and m['fnir@0.01fpir'] < 1e-6
    assert m['n_known_probes'] == 30 and m['n_unknown_probes'] == 10


def test_random_embeddings_near_chance():
    rng = np.random.default_rng(1)
    n_ids, n_gal = 200, 3
    p = np.array([f'{i:03d}' for i in range(n_ids)])
    g = np.repeat(p, n_gal)
    dist = rng.uniform(size=(n_ids, n_ids * n_gal))
    m, _ = split_metrics(dist, p, g, unknown_ids=p[:100])
    assert 40 < m['auroc'] < 60, m['auroc']
    assert 40 < m['eer'] < 60, m['eer']
    assert m['rank1'] < 5 and m['dir@0.01fpir'] < 2 and m['oscr'] < 5


def test_dir_at_fpir_by_hand():
    # 4 known probes (correct match at d=1,2,3 and a wrong match at 0.5), 4 unknown probes at d=1.5,2.5,3.5,4.5
    p = np.array(['a', 'b', 'c', 'd', 'u1', 'u2', 'u3', 'u4'])
    g = np.array(['a', 'b', 'c', 'd', 'x'])
    dist = np.full((8, 5), 9.0)
    dist[0, 0], dist[1, 1], dist[2, 2] = 1, 2, 3
    dist[3, 4] = 0.5                                    # 'd' nearest to the wrong identity
    dist[4:, 4] = [1.5, 2.5, 3.5, 4.5]
    m, c = split_metrics(dist, p, g, unknown_ids=['u1', 'u2', 'u3', 'u4'], fpir=(0.0, 0.25, 0.5))
    assert m['rank1'] == 75                             # a, b, c correct; d wrong
    assert m['dir@0fpir'] == 25                         # tau = 1.5 -> only 'a'
    assert m['dir@0.25fpir'] == 50                      # tau = 2.5 -> a, b
    assert m['dir@0.5fpir'] == 75                       # tau = 3.5 -> a, b, c
    assert np.all(np.diff(c['fpir']) >= 0) and np.all(np.diff(c['dir']) >= 0)
    assert c['fpir'][-1] == 1 and abs(c['dir'][-1] - 0.75) < 1e-12



def test_dir_at_rank_and_su2024_threshold_by_hand():
    # same layout as above; 'd' is at d=0.5 from the wrong identity 'x' and at d=1.2 from its own
    p = np.array(['a', 'b', 'c', 'd', 'u1', 'u2', 'u3', 'u4'])
    g = np.array(['a', 'b', 'c', 'd', 'x'])
    dist = np.full((8, 5), 9.0)
    dist[0, 0], dist[1, 1], dist[2, 2] = 1, 2, 3
    dist[3, 4], dist[3, 3] = 0.5, 1.2
    dist[4:, 4] = [1.5, 2.5, 3.5, 4.5]
    unk = ['u1', 'u2', 'u3', 'u4']
    m, _ = split_metrics(dist, p, g, unk, fpir=(0.0, 0.25, 0.5), dir_ranks=(1, 2))
    assert m['dir@0fpir'] == 25 and m['dir@0fpir_r2'] == 50          # tau = 1.5: a; rank 2 adds d (1.2)
    assert m['dir@0.5fpir'] == 75 and m['dir@0.5fpir_r2'] == 100      # tau = 3.5
    assert m['fnir@0.5fpir_r2'] == 0
    # Su et al.: threshold at the k-th highest non-mated score (k = 1 -> d <= 1.5; k = 2 -> d <= 2.5)
    s, _ = split_metrics(dist, p, g, unk, fpir=(0.0, 0.25, 0.5), dir_ranks=(1, 2), threshold_rule='su2024')
    assert s['dir@0fpir'] == 25 and s['dir@0.25fpir'] == 25 and s['dir@0.5fpir'] == 50
    assert s['dir@0.25fpir_r2'] == 50 and s['dir@0.5fpir_r2'] == 75


def test_fixed_unknown_sets():
    sets = load_su2024_splits('Gait3D')
    assert len(sets) == 50 and {len(set(u)) for u in sets} == {215}
    assert load_su2024_splits('GREW') is None and load_su2024_splits('Gait3D', None) is None
    rng = np.random.default_rng(0)
    p = np.array([f'{i:04d}' for i in range(20)])
    dist = rng.random((20, 20))
    fixed = [p[:5], p[5:9]]
    r = run_protocol(dist, p, p, {**SU2024_CFG, 'name': 'fx', 'unknown_sets': fixed})['fx']
    assert [list(u) for u in r['unknown_ids']] == [list(u) for u in fixed]
    assert [m['n_unknown_probes'] for m in r['metrics']] == [5, 4]

def test_no_leakage_of_unknown_ids():
    dist, p, g = synthetic()
    for unknown in make_splits(p, 0.3, n_splits=5, seed=0):
        _, c = split_metrics(dist, p, g, unknown)
        assert not c['is_known'][np.isin(p, unknown)].any()   # unknown probes never counted as known
        assert c['is_known'][~np.isin(p, unknown)].all()
        assert len(unknown) == 12


def test_splits_reproducible_and_distinct():
    p = np.array([f'{i:03d}' for i in range(100)])
    a, b = make_splits(p, 0.3, 4, seed=0), make_splits(p, 0.3, 4, seed=0)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert len({tuple(s) for s in a}) == 4
    assert list(legacy_split(p)) == [f'{i:03d}' for i in range(90, 100)]


def test_run_protocol_summary():
    dist, p, g = synthetic(noise=1.0)
    res = run_protocol(dist, p, g, {'unknown_ratios': [0.1, 0.5], 'n_splits': 3})
    assert set(res) == {'u10', 'u50', 'legacy10'}
    mean, std = res['u50']['summary']['auroc']
    assert len(res['u50']['metrics']) == 3 and 0 <= mean <= 100 and std >= 0
    assert len(res['legacy10']['metrics']) == 1


def test_subject_templates():
    feats = np.array([[0., 2.], [2., 4.], [10., 10.]])
    t, ids = subject_templates(feats, ['b', 'b', 'a'])
    assert list(ids) == ['a', 'b'] and np.allclose(t, [[10, 10], [1, 3]])
    t, _ = subject_templates(np.array([[3., 0.], [0., 1.]]), ['a', 'a'], normalize=True)
    assert np.allclose(t, [[0.5, 0.5]])                         # mean of unit vectors


def test_localscore_by_hand():
    # templates of subjects a, b, u; sequences a1, a2, b1, u1
    sim_t = np.array([[0.7, 0.5, 0.6],
                      [0.4, 0.3, 0.8]])
    sim_m = np.array([[0.9, 0.2, 0.5, 0.95],
                      [0.1, 0.6, 0.3, 0.99]])
    dist = localscore_dist(sim_t, ['a', 'b', 'u'], sim_m, ['a', 'a', 'b', 'u'], k=1)
    t = -dist(['u'])                                            # u1 must not count as a neighbour
    assert np.allclose(t[:, :2], [[0.7 + 0.9, 0.5], [0.4 + 0.6, 0.3]])
    t2 = -localscore_dist(sim_t, ['a', 'b', 'u'], sim_m, ['a', 'a', 'b', 'u'], k=2)(['u'])
    assert np.allclose(t2[:, :2], [[0.7 + 0.5, 0.5], [0.4 + 0.3, 0.3]])


def test_su2024_protocol():
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(200, 16)) * 3
    probe = centers + rng.normal(scale=0.3, size=centers.shape)
    gallery = np.repeat(centers, 4, 0) + rng.normal(scale=0.3, size=(800, 16))
    lbls = np.array([f'{i:03d}' for i in range(200)])
    g_lbls = np.repeat(lbls, 4)
    res = run_su2024(probe, gallery, lbls, g_lbls, dist_fn)
    r = res['su2024']
    assert set(res) == {'su2024'} and len(r['metrics']) == 50
    assert all(len(u) == 43 for u in r['unknown_ids'])          # round(0.215 * 200)
    assert all(m['n_known_probes'] == 157 for m in r['metrics'])
    assert 'fnir@0.05fpir' in r['median'] and r['median']['auroc'] > 99
    assert np.isclose(r['ci95']['auroc'], 1.96 * r['summary']['auroc'][1] / np.sqrt(50))
    # 1/(1+d) is monotonic in d: same metrics as ranking on raw distances
    templates, t_lbls = subject_templates(gallery, g_lbls)
    raw = run_protocol(euclidean(probe, templates), lbls, t_lbls, {**SU2024_CFG, 'name': 'raw'})['raw']
    assert all(np.isclose(a[k], b[k]) for a, b in zip(raw['metrics'], r['metrics']) for k in a)
    for sim, k, key in [('cos', 0, 'su2024_cos'), ('euc', 1, 'su2024_ls1'), ('cos', 1, 'su2024_cos_ls1')]:
        rr = run_su2024(probe, gallery, lbls, g_lbls, dist_fn, similarity=sim, localscore_k=k)[key]
        assert rr['median']['auroc'] > 99
        if k:  # LocalScore only boosts the top subject: identification unchanged
            base = run_su2024(probe, gallery, lbls, g_lbls, dist_fn, similarity=sim)
            base = next(iter(base.values()))
            assert [m['rank1'] for m in rr['metrics']] == [m['rank1'] for m in base['metrics']]


if __name__ == '__main__':
    tests = [v for k, v in dict(globals()).items() if k.startswith('test_')]
    for t in tests:
        t()
        print(f'OK  {t.__name__}')
    print(f'{len(tests)} tests OK')
