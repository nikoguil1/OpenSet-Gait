import numpy as np
import pickle
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

def part_based_euc_dist(x, y):
    N, C, P = x.shape
    M = y.shape[0]
    dist = np.zeros((N, M))
    for i in range(P):
        _x = x[:, :, i]
        _y = y[:, :, i]
        _x2 = np.sum(_x**2, axis=1, keepdims=True)
        _y2 = np.sum(_y**2, axis=1, keepdims=0)
        _dist = _x2 + _y2 - 2 * np.dot(_x, _y.T)
        dist += np.sqrt(np.maximum(_dist, 0))
    return dist / P

def calculate_far_frr(dist_matrix, probe_lbls, known_ids):
    min_distances = np.min(dist_matrix, axis=1)
    is_known_mask = np.isin(probe_lbls, known_ids)
    known_dists = min_distances[is_known_mask]
    intruder_dists = min_distances[~is_known_mask]

    thresholds = np.linspace(min_distances.min(), min_distances.max(), 1000)
    fars = []
    frrs = []

    for t in thresholds:
        far = np.mean(intruder_dists < t)
        frr = np.mean(known_dists > t)
        fars.append(far)
        frrs.append(frr)

    fars = np.array(fars)
    frrs = np.array(frrs)

    diff = np.abs(fars - frrs)
    eer_idx = np.argmin(diff)
    eer_threshold = thresholds[eer_idx]
    eer_value = (fars[eer_idx] + frrs[eer_idx]) / 2
    
    return eer_value, eer_threshold, thresholds, fars, frrs

# Load data
pkl_file = "roc_data/casiab_all_embeddings.pkl"
print(f"Loading data from {pkl_file}...")
with open(pkl_file, "rb") as f:
    data = pickle.load(f)

feature = data['features']
label = data['lbls']
seq_type = data['types']

unique_test_ids = np.unique(label)
num_intruders = 5 # 10% of 50
known_ids = unique_test_ids[:-num_intruders]

gallery_mask_nm = np.isin(seq_type, ['nm-01', 'nm-02', 'nm-03', 'nm-04'])
gallery_known_mask = gallery_mask_nm & np.isin(label, known_ids)
gallery_features = feature[gallery_known_mask, :]

conditions = {
    'NM': ['nm-05', 'nm-06'],
    'BG': ['bg-01', 'bg-02'],
    'CL': ['cl-01', 'cl-02']
}

results = {}
colors = {'NM': ('r', 'darkred'), 'BG': ('g', 'darkgreen'), 'CL': ('b', 'darkblue')}

plt.figure(figsize=(12, 8))

for cond, seqs in conditions.items():
    print(f"\nProcessing condition {cond}...")
    probe_mask = np.isin(seq_type, seqs)
    probe_features = feature[probe_mask, :]
    probe_lbls = label[probe_mask]
    
    print(f"Calculating distances for {cond} (Probe size: {probe_features.shape[0]})...")
    dist_matrix = part_based_euc_dist(probe_features, gallery_features)
    
    eer, th, thrsh, far, frr = calculate_far_frr(dist_matrix, probe_lbls, known_ids)
    print(f"{cond} EER: {eer*100:.2f}% at threshold: {th:.4f}")
    
    # Plot curves
    color_far, color_frr = colors[cond]
    plt.plot(thrsh, far * 100, color=color_far, linestyle='-', label=f'{cond} FAR')
    plt.plot(thrsh, frr * 100, color=color_frr, linestyle='--', label=f'{cond} FRR')
    
    # Plot EER point
    plt.plot(th, eer * 100, 'ko', markersize=6)
    plt.annotate(f"{cond} EER={eer*100:.2f}%", xy=(th, eer*100), xytext=(th+0.5, eer*100+2))

plt.title('FAR vs FRR for NM, BG, and CL Conditions (CASIA-B)')
plt.xlabel('Distance Threshold')
plt.ylabel('Error Rate (%)')
plt.xlim(left=0)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)

output_png = "roc_data/far_frr_plot_casiab_nm_bg_cl.png"
plt.savefig(output_png, bbox_inches='tight')
print(f"\nPlot saved to {output_png}")
