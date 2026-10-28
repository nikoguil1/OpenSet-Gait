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

def get_eer_and_plot_data(pkl_path):
    print(f"Loading data from {pkl_path}...")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    probe_features = data['probe_features']
    gallery_features = data['gallery_features']
    probe_lbls = np.asarray(data['probe_lbls'])
    gallery_lbls = np.asarray(data['gallery_lbls'])

    # Define known vs intruder IDs (90/10 split)
    unique_test_ids = np.unique(np.concatenate([probe_lbls, gallery_lbls]))
    num_intruders = int(len(unique_test_ids) * 0.10)
    known_ids = unique_test_ids[:-num_intruders]

    # Filter gallery to only include known IDs
    gallery_known_mask = np.isin(gallery_lbls, known_ids)
    gallery_known_features = gallery_features[gallery_known_mask]

    print("Calculating minimum distances...")
    dist_matrix = part_based_euc_dist(probe_features, gallery_known_features)
    min_distances = np.min(dist_matrix, axis=1)

    # Separate distances
    is_known_mask = np.isin(probe_lbls, known_ids)
    known_dists = min_distances[is_known_mask]
    intruder_dists = min_distances[~is_known_mask]

    # Define threshold range
    thresholds = np.linspace(min_distances.min(), min_distances.max(), 500)
    fars = []
    frrs = []

    for t in thresholds:
        far = np.mean(intruder_dists < t)
        frr = np.mean(known_dists > t)
        fars.append(far)
        frrs.append(frr)

    fars = np.array(fars)
    frrs = np.array(frrs)

    # Find EER
    diff = np.abs(fars - frrs)
    eer_idx = np.argmin(diff)
    eer_threshold = thresholds[eer_idx]
    eer_value = (fars[eer_idx] + frrs[eer_idx]) / 2
    
    return eer_value, eer_threshold, thresholds, fars, frrs

# Files to compare
file_old = "roc_data/embeddings_Gait3D_DeepGaitV2_triplet_margin0.2.pkl"
file_new = "roc_data/embeddings_Gait3D_DeepGaitV2_triplet_margin-var.pkl"

# Process New Model
print("\n" + "="*50)
print("PROCESSING NEW MODEL (margin-var)")
print("="*50)
eer_new, th_new, t_new, fars_new, frrs_new = get_eer_and_plot_data(file_new)

# Process Old Model
print("\n" + "="*50)
print("PROCESSING OLD MODEL (margin 0.2)")
print("="*50)
eer_old, th_old, t_old, fars_old, frrs_old = get_eer_and_plot_data(file_old)

# Plotting for New Model
plt.figure(figsize=(10, 6))
plt.plot(t_new, fars_new * 100, 'r', label='FAR (False Acceptance Rate)', lw=2)
plt.plot(t_new, frrs_new * 100, 'b', label='FRR (False Rejection Rate)', lw=2)

plt.plot(th_new, eer_new * 100, 'ko', markersize=8, label=f'EER: {eer_new*100:.2f}%')
plt.axvline(th_new, color='gray', linestyle='--', alpha=0.5)
plt.axhline(eer_new * 100, color='gray', linestyle='--', alpha=0.5)

plt.title('FAR vs FRR for Intruder Detection (Gait3D - New Model margin-var)')
plt.xlabel('Distance Threshold')
plt.ylabel('Error Rate (%)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)

plt.annotate(f'EER={eer_new*100:.2f}%\nTh={th_new:.2f}', 
             xy=(th_new, eer_new * 100), 
             xytext=(th_new + 0.5, eer_new * 100 + 10),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

output_png = "roc_data/far_frr_plot_gait3d_deepgaitV2-triplet_margin-var.png"
plt.savefig(output_png, bbox_inches='tight')
print(f"\nPlot saved to {output_png}")

print("\n" + "="*50)
print("COMPARISON RESULTS")
print("="*50)
print(f"Old Model (margin 0.2):  EER = {eer_old*100:.2f}% at Threshold: {th_old:.4f}")
print(f"New Model (margin-var):  EER = {eer_new*100:.2f}% at Threshold: {th_new:.4f}")
diff = eer_new - eer_old
if diff < 0:
    print(f"Improvement: {-diff*100:.2f}% lower error.")
else:
    print(f"Regression: {diff*100:.2f}% higher error.")
print("="*50)
