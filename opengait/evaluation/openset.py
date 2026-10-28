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
    'output_dir': 'roc_data',
}

# Su et al., ECCV 2024 (followed by LocalScore, arXiv 2602.01012): 21.5% of the test
# subjects are non-mated, 50 random sets, one mean template per gallery subject,
# reported as median +- std. Only the split/aggregation is reproduced; the distance
# is still the evaluator's metric (their similarity function is not specified).
SU2024_CFG = {
    'unknown_ratios': [0.215],
    'n_splits': 50,
    'seed': 0,
    'fpir': [0.01, 0.05],
    'far': [0.01, 0.001],
    'legacy_split': False,
}


def make_splits(probe_lbls, unknown_ratio, n_splits, seed):
    """Random sets of unknown IDs, drawn from the identities present in the probe set."""
    ids = np.unique(probe_lbls)
    n_unknown = int(round(len(ids) * unknown_ratio))
    if not 0 < n_unknown < len(ids):
        raise ValueError(f'unknown_ratio={unknown_ratio} leaves no known or no unknown IDs')
    rng = np.random.default_rng(seed)
    return [np.sort(rng.choice(ids, n_unknown, replace=False)) for _ in range(n_splits)]


def legacy_split(labels):
    """Split used before this protocol: the last 10% of the sorted test IDs are intruders."""
    ids = np.unique(labels)
    return ids[-int(len(ids) * 0.10):]


def eer_from_roc(fpr, tpr):
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return (fpr[i] + fnr[i]) / 2


def split_metrics(dist, probe_lbls, gallery_lbls, unknown_ids, fpir=(0.01, 0.1), far=(0.01, 0.001)):
    """Metrics for one split.

    dist: [n_probe, n_gallery] distances against the FULL gallery.
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

    # Open-set identification: accept a probe iff d_min < tau
    for f in fpir:
        k = int(np.floor(f * n_unknown))  # max unknown probes allowed to be accepted
        tau = d_unknown[k] if k < n_unknown else np.inf
        dir_ = (correct & (d_min < tau)).sum() / n_known
        m[f'dir@{f:g}fpir'] = dir_ * 100
        m[f'fnir@{f:g}fpir'] = (1 - dir_) * 100

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


def subject_templates(features, labels):
    """Mean feature per subject: ([n_subjects, ...] templates, [n_subjects] sorted labels)."""
    labels = np.asarray(labels)
    ids, inv = np.unique(labels, return_inverse=True)
    feats = np.asarray(features, dtype=np.float32)
    templates = np.zeros((len(ids),) + feats.shape[1:], dtype=np.float32)
    np.add.at(templates, inv, feats)
    counts = np.bincount(inv).astype(np.float32).reshape((-1,) + (1,) * (feats.ndim - 1))
    return templates / counts, ids


def run_protocol(dist, probe_lbls, gallery_lbls, cfg=None):
    """Evaluate every unknown ratio (and the legacy split) on a precomputed distance matrix.

    Returns {split_name: {'unknown_ids': [...], 'metrics': [...], 'curves': [...], 'summary': {...}}}.
    """
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    probe_lbls, gallery_lbls = np.asarray(probe_lbls), np.asarray(gallery_lbls)
    protocols = {cfg.get('name') or f'u{int(round(r * 100))}': make_splits(probe_lbls, r, cfg['n_splits'], cfg['seed'])
                 for r in cfg['unknown_ratios']}
    if cfg['legacy_split']:
        protocols['legacy10'] = [legacy_split(np.concatenate([probe_lbls, gallery_lbls]))]

    out = {}
    for name, splits in protocols.items():
        res = [split_metrics(dist, probe_lbls, gallery_lbls, u, cfg['fpir'], cfg['far']) for u in splits]
        metrics = [r[0] for r in res]
        out[name] = {'unknown_ids': splits, 'metrics': metrics,
                     'curves': [r[1] for r in res], 'summary': aggregate(metrics),
                     'median': {k: float(np.median([m[k] for m in metrics])) for k in metrics[0]}}
    return out


def run_su2024(probe_features, gallery_features, probe_lbls, gallery_lbls, dist_fn, cfg=None):
    """SU2024_CFG protocol on per-subject mean gallery templates.

    dist_fn(probe_features, templates) -> [n_probe, n_subjects] numpy distances.
    Templates do not depend on the split: removing a subject drops its template.
    """
    templates, template_lbls = subject_templates(gallery_features, gallery_lbls)
    dist = dist_fn(probe_features, templates)
    return run_protocol(dist, probe_lbls, template_lbls, {**SU2024_CFG, 'name': 'su2024', **(cfg or {})})


def format_summary(results):
    """One line per split protocol, mean +- std (median-based for su2024)."""
    lines = []
    for name, r in results.items():
        stats = {k: (r['median'][k], s) for k, (_, s) in r['summary'].items()} if name == 'su2024' else r['summary']
        cells = [f'{k}={m:.2f}' + (f'±{s:.2f}' if len(r['metrics']) > 1 else '')
                 for k, (m, s) in stats.items() if not k.startswith('n_')]
        tag = 'median±std' if name == 'su2024' else 'mean±std'
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
