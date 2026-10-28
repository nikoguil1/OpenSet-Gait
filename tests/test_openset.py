"""Unit tests for opengait/evaluation/openset.py on synthetic distances.

Run: python tests/test_openset.py   (or pytest tests/)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'opengait'))
from evaluation.openset import legacy_split, make_splits, run_protocol, split_metrics  # noqa: E402


def euclidean(x, y):
    return np.sqrt(np.maximum((x ** 2).sum(1)[:, None] + (y ** 2).sum(1)[None] - 2 * x @ y.T, 0))


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


if __name__ == '__main__':
    tests = [v for k, v in dict(globals()).items() if k.startswith('test_')]
    for t in tests:
        t()
        print(f'OK  {t.__name__}')
    print(f'{len(tests)} tests OK')
