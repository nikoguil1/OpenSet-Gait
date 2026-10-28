import numpy as np
import pickle
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm

# Path to the data
pkl_file = "roc_data/embeddings_Gait3D_DeepGaitV2_triplet_margin0.2.pkl"

if not os.path.exists(pkl_file):
    print(f"Error: {pkl_file} not found.")
    exit(1)

print(f"Loading data from {pkl_file}...")
with open(pkl_file, "rb") as f:
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

# Part-based distance calculation (matches OpenGait's cuda_dist)
def part_based_euc_dist(x, y):
    # x: [N, C, P], y: [M, C, P]
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

print("Calculating minimum distances from Probe to Gallery (Known IDs)...")
dist_matrix = part_based_euc_dist(probe_features, gallery_known_features)
min_distances = np.min(dist_matrix, axis=1)

# Separate distances into Known Probes and Intruder Probes
is_known_mask = np.isin(probe_lbls, known_ids)
known_dists = min_distances[is_known_mask]
intruder_dists = min_distances[~is_known_mask]

print(f"Number of Known samples: {len(known_dists)}")
print(f"Number of Intruder samples: {len(intruder_dists)}")

# Model with Gaussians
mu_known, std_known = norm.fit(known_dists)
mu_intruder, std_intruder = norm.fit(intruder_dists)

print(f"Known: mu={mu_known:.4f}, std={std_known:.4f}")
print(f"Intruder: mu={mu_intruder:.4f}, std={std_intruder:.4f}")

# Plotting
plt.figure(figsize=(12, 7))

# Range for X axis
xmin = min(known_dists.min(), intruder_dists.min()) * 0.8
xmax = max(known_dists.max(), intruder_dists.max()) * 1.2
x = np.linspace(xmin, xmax, 1000)

# Gaussian PDFs
p_known = norm.pdf(x, mu_known, std_known)
p_intruder = norm.pdf(x, mu_intruder, std_intruder)

plt.plot(x, p_known, 'b', linewidth=2, label=f'Known Identities ($\mu$={mu_known:.2f}, $\sigma$={std_known:.2f})')
plt.fill_between(x, p_known, alpha=0.2, color='b')

plt.plot(x, p_intruder, 'r', linewidth=2, label=f'Intruders ($\mu$={mu_intruder:.2f}, $\sigma$={std_intruder:.2f})')
plt.fill_between(x, p_intruder, alpha=0.2, color='r')

# Calculate Intersection (Theoretical EER point)
# Simplified: EER at the intersection of the two Gaussians if they had equal priors
# For visual reference only
plt.title('Gaussian Modeling of Minimum Distances (Known vs Intruders)')
plt.xlabel('Minimum Distance to Gallery')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

output_png = "roc_data/gaussian_modeling_distances.png"
plt.savefig(output_png, bbox_inches='tight')
print(f"Plot saved to {output_png}")
