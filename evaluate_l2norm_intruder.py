import numpy as np
import pickle
import os

def part_based_euc_dist(x, y):
    """
    Computes Euclidean distance for part-based representations.
    x shape: (N, C, P)
    y shape: (M, C, P)
    Returns: (N, M) distance matrix.
    """
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

def normalize_features(features):
    """
    L2 Normalizes the features along the channel dimension (C).
    Input shape: (N, C, P)
    """
    norms = np.linalg.norm(features, ord=2, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    return features / norms

def calculate_metrics(min_distances, probe_lbls, known_ids):
    is_known_mask = np.isin(probe_lbls, known_ids)
    known_dists = min_distances[is_known_mask]
    intruder_dists = min_distances[~is_known_mask]

    thresholds = np.linspace(min_distances.min(), min_distances.max(), 1000)
    fars = []
    frrs = []

    for t in thresholds:
        fars.append(np.mean(intruder_dists < t))
        frrs.append(np.mean(known_dists > t))

    fars = np.array(fars)
    frrs = np.array(frrs)
    diff = np.abs(fars - frrs)
    eer_idx = np.argmin(diff)
    
    eer_value = (fars[eer_idx] + frrs[eer_idx]) / 2
    eer_threshold = thresholds[eer_idx]
    
    # TAR at 1% FAR
    sorted_impostors = np.sort(intruder_dists)
    idx = int(0.01 * len(sorted_impostors))
    if idx >= len(sorted_impostors): idx = len(sorted_impostors) - 1
    
    tar_threshold = sorted_impostors[idx]
    tar_1far = np.mean(known_dists < tar_threshold)
    actual_far = np.mean(intruder_dists < tar_threshold)

    return eer_value, eer_threshold, tar_1far, actual_far, tar_threshold

# Load Gait3D Margin-Var Embeddings
pkl_path = "roc_data/embeddings_Gait3D_DeepGaitV2_triplet_margin-var.pkl"
print(f"Loading data from {pkl_path}...")
with open(pkl_path, "rb") as f:
    data = pickle.load(f)

probe_features = data['probe_features']
gallery_features = data['gallery_features']
probe_lbls = np.asarray(data['probe_lbls'])
gallery_lbls = np.asarray(data['gallery_lbls'])

unique_test_ids = np.unique(np.concatenate([probe_lbls, gallery_lbls]))
num_intruders = int(len(unique_test_ids) * 0.10)
known_ids = unique_test_ids[:-num_intruders]

gallery_known_mask = np.isin(gallery_lbls, known_ids)
gallery_known_features = gallery_features[gallery_known_mask]

# L2 Normalize
print("Applying L2 Normalization to embeddings...")
probe_norm = normalize_features(probe_features)
gallery_norm = normalize_features(gallery_known_features)

print("Calculating Distances (Cosine proxy via Normalized Euclidean)...")
# Computing Euclidean on L2-normalized vectors perfectly orders Cosine distance mathematically.
dist_matrix = part_based_euc_dist(probe_norm, gallery_norm)

# Minimum distance (Nearest Neighbor) to the known gallery
min_distances = np.min(dist_matrix, axis=1)

print("Calculating Metrics...")
eer, eer_th, tar_1far, far, tar_th = calculate_metrics(min_distances, probe_lbls, known_ids)

print("="*50)
print("INTRUDER DETECTION METRICS (L2 NORM + COSINE/NN)")
print("="*50)
print(f"Dataset: Gait3D | Model: DeepGaitV2 (margin=var)")
print(f"  EER          : {eer*100:.2f}% (Threshold: {eer_th:.4f})")
print(f"  TAR @ 1% FAR : {tar_1far*100:.2f}% (Actual FAR: {far*100:.2f}%, Threshold: {tar_th:.4f})")
print("="*50)
