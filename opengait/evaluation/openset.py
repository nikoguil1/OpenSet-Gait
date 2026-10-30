"""Open-set evaluation protocol.

A subset of the test identities is declared *unknown*: their sequences are
removed from the gallery, while all probes are kept. Every metric is computed
for several random known/unknown splits and reported as mean +- std.

Metrics (per split):
  - closed-set Rank-1 of the known probes against the reduced gallery
  - verification over all probe x gallery pairs: EER, TAR@FAR
  - open-set identification on the nearest-neighbour distance d_min:
    DIR@FPIR (= 1 - FNIR@FPIR), OSCR and AUROC (known vs unknown probes)

References: Grother et al., FRVT 1:N (DIR/FPIR/FNIR); Dhamija et al., NeurIPS 2018
(OSCR); Su et al., "Open-Set Biometrics: Beyond Good Closed-Set Models", ECCV 2024.
"""
import json
import os
import pickle

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

DEFAULT_CFG = {
    'unknown_ratios': [0.1, 0.3, 0.5],
    'n_splits': 10,
    'seed': 0,
    'fpir': [0.01, 0.1],
    'far': [0.01, 0.001],
    'legacy_split': True,  # last 10% of sorted IDs as intruders (pre-protocol code)
    'su2024': True,  # also run the SU2024_CFG protocol on per-subject mean templates
    'su2024_similarity': ['euc', 'cos'],  # template scoring: 1/(1+d) (Su eq. 6) or cosine (LocalScore eq. 1)
    'su2024_localscore_k': [0, 1],  # 0 = plain template score; k>0 adds LocalScore with the k-th NN
    'su2024_splits': 'official',  # Su et al.'s non-mated sets when bundled (SU2024_SPLITS), else random; or a json path
    'dir_ranks': [1],  # DIR/FNIR@FPIR also at rank R > 1 (keys *_r{R})
    'threshold_rule': 'strict',  # 'su2024': threshold at the k-th highest non-mated score, as in Su et al.'s code
    'output_dir': 'roc_data',
}

# Su et al., ECCV 2024 (followed by LocalScore, arXiv 2602.01012): 21.5% of the test
# subjects are non-mated, 50 random sets, one mean template per gallery subject,
# reported as median +- std (LocalScore: mean +- 95% CI, i.e. 1.96 std / sqrt(n)).
# Su et al. score gait with s = 1/(1+d) on Euclidean distances; LocalScore writes the
# template score as the cosine to the mean of L2-normalised gallery samples.
# Su et al. report FNIR@FPIR at rank R = 20 (sec. 4.1; compute_dir_far(ranks=[20]) in their code)
# and the closed-set Rank-1 on the full gallery, not against the templates.
SU2024_CFG = {
    'unknown_ratios': [0.215],
    'n_splits': 50,
    'seed': 0,
    'fpir': [0.01, 0.05],
    'far': [0.01, 0.001],
    'legacy_split': False,
    'dir_ranks': [1, 20],
    'threshold_rule': 'su2024',
}

# Non-mated subject sets released by Su et al. (datasets/Gait3D/openset_nonmated.npy, converted to IDs)
SU2024_SPLITS = {'Gait3D': 'datasets/Gait3D/su2024_nonmated_ids.json'}


def make_splits(probe_lbls, unknown_ratio, n_splits, seed):
    """Random sets of unknown IDs, drawn from the identities present in the probe set."""
    ids = np.unique(probe_lbls)
    n_unknown = int(round(len(ids) * unknown_ratio))
    if not 0 < n_unknown < len(ids):
        raise ValueError(f'unknown_ratio={unknown_ratio} leaves no known or no unknown IDs')
    rng = np.random.default_rng(seed)
    return [np.sort(rng.choice(ids, n_unknown, replace=False)) for _ in range(n_splits)]


def load_su2024_splits(dataset, setting='official'):
    """Non-mated ID sets for the su2024 protocol, or None for random splits.

    setting: 'official' (bundled file of SU2024_SPLITS for this dataset, if any), a json path, or None.
    """
    path = SU2024_SPLITS.get(dataset) if setting == 'official' else setting
    if not path:
        return None
    with open(path) as f:
        return [np.asarray(s) for s in json.load(f)['nonmated_ids']]


def legacy_split(labels):
    """Split used before this protocol: the last 10% of the sorted test IDs are intruders."""
    ids = np.unique(labels)
    return ids[-int(len(ids) * 0.10):]


def eer_from_roc(fpr, tpr):
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return (fpr[i] + fnr[i]) / 2


def split_metrics(dist, probe_lbls, gallery_lbls, unknown_ids, fpir=(0.01, 0.1), far=(0.01, 0.001),
                  dir_ranks=(1,), threshold_rule='strict'):
    """Metrics for one split.

    dist: [n_probe, n_gallery] distances against the FULL gallery.
    dir_ranks: DIR@FPIR at rank R > 1 accepts a known probe iff its genuine distance passes the
      threshold and its identity is among the R nearest gallery identities (Su et al.).
    threshold_rule: with k = floor(FPIR * n_unknown) unknown probes allowed, 'strict' accepts
      d < (k+1)-th smallest unknown d_min; 'su2024' accepts d <= k-th smallest (d < smallest if k = 0).
    Returns (metrics dict, curves dict).
    """
    keep = ~np.isin(gallery_lbls, unknown_ids)
    dist, g_lbls = dist[:, keep], gallery_lbls[keep]
    # A probe is known iff its identity is still enrolled in the reduced gallery
    is_known = np.isin(probe_lbls, g_lbls)
    if is_known.all() or not is_known.any():
        raise ValueError('split needs both known and unknown probes')

    nn = dist.argmin(1)
    d_min = dist[np.arange(len(dist)), nn]
    correct = (g_lbls[nn] == probe_lbls) & is_known

    d_known, d_unknown = d_min[is_known], np.sort(d_min[~is_known])
    n_known, n_unknown = is_known.sum(), len(d_unknown)
    m = {'rank1': correct[is_known].mean() * 100,
         'n_known_probes': int(n_known), 'n_unknown_probes': int(n_unknown)}

    # Verification over all pairs (genuine = same ID)
    genuine = (probe_lbls[:, None] == g_lbls[None, :]).ravel()
    fpr, tpr, _ = roc_curve(genuine, -dist.ravel())
    m['eer'] = eer_from_roc(fpr, tpr) * 100
    for f in far:
        m[f'tar@{f:g}far'] = np.interp(f, fpr, tpr) * 100

    if any(r > 1 for r in dir_ranks):
        # distance to the genuine identity and its rank among the gallery identities
        ids, inv = np.unique(g_lbls, return_inverse=True)
        d_id = np.full((len(ids), len(dist)), np.inf)
        np.minimum.at(d_id, inv, dist.T)
        d_id = d_id.T  # [n_probe, n_ids]
        own = np.searchsorted(ids, probe_lbls).clip(max=len(ids) - 1)
        d_gen = np.where(is_known, d_id[np.arange(len(dist)), own], np.inf)
        rank_gen = (d_id < d_gen[:, None]).sum(1) + 1

    # Open-set identification: accept a probe iff d_min < tau
    for f in fpir:
        k = int(np.floor(f * n_unknown))  # max unknown probes allowed to be accepted
        if threshold_rule == 'su2024' and k > 0:
            tau = np.nextafter(d_unknown[k - 1], np.inf)
        else:
            tau = d_unknown[k] if k < n_unknown else np.inf
        for r in dir_ranks:
            hit = correct & (d_min < tau) if r == 1 else is_known & (d_gen < tau) & (rank_gen <= r)
            dir_ = hit.sum() / n_known
            sfx = '' if r == 1 else f'_r{r}'
            m[f'dir@{f:g}fpir{sfx}'] = dir_ * 100
            m[f'fnir@{f:g}fpir{sfx}'] = (1 - dir_) * 100

    # Full DIR vs FPIR curve (accept iff d_min <= tau) and OSCR = area under it
    taus = np.unique(d_min)
    fpir_curve = np.searchsorted(d_unknown, taus, side='right') / n_unknown
    d_correct = np.sort(d_min[correct])
    dir_curve = np.searchsorted(d_correct, taus, side='right') / n_known
    fpir_curve = np.concatenate([[0.], fpir_curve, [1.]])
    dir_curve = np.concatenate([[0.], dir_curve, [dir_curve[-1]]])
    m['oscr'] = np.trapezoid(dir_curve, fpir_curve) * 100
    m['auroc'] = roc_auc_score(is_known, -d_min) * 100

    curves = {'fpir': fpir_curve, 'dir': dir_curve, 'd_min': d_min, 'is_known': is_known,
              'correct': correct, 'verif_fpr': fpr[::max(1, len(fpr) // 2000)],
              'verif_tpr': tpr[::max(1, len(tpr) // 2000)]}
    return m, curves


def aggregate(split_results):
    keys = split_results[0].keys()
    return {k: (float(np.mean([r[k] for r in split_results])),
                float(np.std([r[k] for r in split_results]))) for k in keys}


def subject_templates(features, labels, normalize=False):
    """Mean feature per subject: ([n_subjects, ...] templates, [n_subjects] sorted labels).

    normalize: L2-normalise each sample over axis 1 (channels, i.e. per part as in
    cuda_dist(metric='cos')) before averaging.
    """
    labels = np.asarray(labels)
    ids, inv = np.unique(labels, return_inverse=True)
    feats = np.asarray(features, dtype=np.float32)
    if normalize:
        feats = feats / np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-12)
    templates = np.zeros((len(ids),) + feats.shape[1:], dtype=np.float32)
    np.add.at(templates, inv, feats)
    counts = np.bincount(inv).astype(np.float32).reshape((-1,) + (1,) * (feats.ndim - 1))
    return templates / counts, ids


def to_similarity(dist, similarity):
    """cuda_dist output -> similarity score: 1/(1+d) for 'euc', 1 - d (= cosine) for 'cos'."""
    return 1 / (1 + dist) if similarity == 'euc' else 1 - dist


def localscore_dist(sim_t, t_lbls, sim_m, m_lbls, k=1):
    """LocalScore (arXiv 2602.01012, eq. 4) as a per-split distance for run_protocol.

    sim_t: [n_probe, n_subjects] template scores; sim_m: [n_probe, n_media] per-sequence scores.
    For each split the k-th largest per-sequence score over the REDUCED gallery (no labels
    used) is added to the top-scoring enrolled subject only. Returns f(unknown_ids) -> -scores.
    """
    t_lbls, m_lbls = np.asarray(t_lbls), np.asarray(m_lbls)

    def dist(unknown_ids):
        t = sim_t.copy()
        cols = np.flatnonzero(~np.isin(t_lbls, unknown_ids))
        media = sim_m[:, ~np.isin(m_lbls, unknown_ids)]
        s_k = -np.partition(-media, k - 1, axis=1)[:, k - 1]
        top = cols[t[:, cols].argmax(1)]
        t[np.arange(len(t)), top] += s_k
        return -t
    return dist


def run_protocol(dist, probe_lbls, gallery_lbls, cfg=None):
    """Evaluate every unknown ratio (and the legacy split) on a precomputed distance matrix.

    dist may also be a callable f(unknown_ids) -> distance matrix, for split-dependent scores.
    cfg['unknown_sets'] (list of ID arrays) replaces the random splits of the single unknown ratio.

    Returns {split_name: {'unknown_ids': [...], 'metrics': [...], 'curves': [...], 'summary': {...}}}.
    """
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    probe_lbls, gallery_lbls = np.asarray(probe_lbls), np.asarray(gallery_lbls)
    if cfg.get('unknown_sets') is not None:
        protocols = {cfg.get('name') or 'custom': list(cfg['unknown_sets'])}
    else:
        protocols = {cfg.get('name') or f'u{int(round(r * 100))}': make_splits(probe_lbls, r, cfg['n_splits'], cfg['seed'])
                     for r in cfg['unknown_ratios']}
    if cfg['legacy_split']:
        protocols['legacy10'] = [legacy_split(np.concatenate([probe_lbls, gallery_lbls]))]

    out = {}
    for name, splits in protocols.items():
        res = [split_metrics(dist(u) if callable(dist) else dist, probe_lbls, gallery_lbls, u,
                             cfg['fpir'], cfg['far'], cfg['dir_ranks'], cfg['threshold_rule']) for u in splits]
        metrics = [r[0] for r in res]
        summary = aggregate(metrics)
        out[name] = {'unknown_ids': splits, 'metrics': metrics,
                     'curves': [r[1] for r in res], 'summary': summary,
                     'median': {k: float(np.median([m[k] for m in metrics])) for k in metrics[0]},
                     'ci95': {k: 1.96 * s / np.sqrt(len(metrics)) for k, (_, s) in summary.items()}}
    return out


def run_su2024(probe_features, gallery_features, probe_lbls, gallery_lbls, dist_fn,
               similarity='euc', localscore_k=0, cfg=None):
    """SU2024_CFG protocol on per-subject mean gallery templates.

    dist_fn(x, y, metric) -> [len(x), len(y)] numpy distances (cuda_dist convention).
    similarity: 'euc' (mean of raw features, s = 1/(1+d)) or 'cos' (mean of L2-normalised
    features, cosine). localscore_k > 0 adds LocalScore with the k-th nearest sequence.
    Templates do not depend on the split: removing a subject drops its template.
    Its 'rank1' is against the templates; Su et al. report the closed-set Rank-1 instead.
    Result key: su2024[_cos][_ls{k}].
    """
    templates, template_lbls = subject_templates(gallery_features, gallery_lbls, normalize=similarity == 'cos')
    sim_t = to_similarity(dist_fn(probe_features, templates, similarity), similarity)
    if localscore_k:
        sim_m = to_similarity(dist_fn(probe_features, gallery_features, similarity), similarity)
        dist = localscore_dist(sim_t, template_lbls, sim_m, gallery_lbls, localscore_k)
    else:
        dist = -sim_t
    name = 'su2024' + ('_cos' if similarity == 'cos' else '') + (f'_ls{localscore_k}' if localscore_k else '')
    return run_protocol(dist, probe_lbls, template_lbls, {**SU2024_CFG, 'name': name, **(cfg or {})})


def format_summary(results):
    """One line per split protocol, mean +- std (su2024*: median +- std and mean +- 95% CI)."""
    lines = []
    for name, r in results.items():
        views = [('mean±std', r['summary'])]
        if name.startswith('su2024'):
            views = [('median±std', {k: (r['median'][k], s) for k, (_, s) in r['summary'].items()}),
                     ('mean±ci95', {k: (m, r['ci95'][k]) for k, (m, _) in r['summary'].items()})]
        for tag, stats in views:
            cells = [f'{k}={m:.2f}' + (f'±{s:.2f}' if len(r['metrics']) > 1 else '')
                     for k, (m, s) in stats.items() if not k.startswith('n_')]
            lines.append(f'[{name}, {len(r["metrics"])} split(s), {tag}] ' + ' '.join(cells))
    return lines


def to_scalars(results, prefix='scalar/openset'):
    return {f'{prefix}/{name}/{k}': m for name, r in results.items()
            for k, (m, _) in r['summary'].items() if not k.startswith('n_')}


def save_results(results, dataset, save_name, output_dir, **arrays):
    """Write {output_dir}/{dataset}_{save_name}_openset.pkl; returns the path."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f'{dataset}_{save_name}_openset.pkl')
    with open(path, 'wb') as f:
        pickle.dump({'dataset': dataset, 'save_name': save_name, 'results': results, **arrays}, f)
    return path
