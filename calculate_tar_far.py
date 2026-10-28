import numpy as np
import pickle
import os

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

def calculate_tar_at_far(pkl_path, target_far=0.01):
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

    dist_matrix = part_based_euc_dist(probe_features, gallery_known_features)
    min_distances = np.min(dist_matrix, axis=1)

    is_known_mask = np.isin(probe_lbls, known_ids)
    known_dists = min_distances[is_known_mask]
    intruder_dists = min_distances[~is_known_mask]

    # Calculate TAR at specific FAR
    # A sample is accepted if dist < threshold
    # FAR = mean(intruder_dists < threshold)
    # TAR = mean(known_dists < threshold)
    
    thresholds = np.sort(intruder_dists)
    idx = int(target_far * len(thresholds))
    if idx >= len(thresholds): idx = len(thresholds) - 1
    
    # Threshold that gives exactly (or closest to) target_far
    threshold = thresholds[idx]
    
    tar = np.mean(known_dists < threshold)
    actual_far = np.mean(intruder_dists < threshold)
    
    return tar, actual_far, threshold

file_old = "roc_data/embeddings_Gait3D_DeepGaitV2_triplet_margin0.2.pkl"
file_new = "roc_data/embeddings_Gait3D_DeepGaitV2_triplet_margin-var.pkl"

print("="*50)
print("INTRUDER DETECTION METRICS (TAR @ 1% FAR)")
print("="*50)

tar_new, far_new, th_new = calculate_tar_at_far(file_new, 0.01)
print(f"New Model (margin-var):")
print(f"  TAR @ 1% FAR: {tar_new*100:.2f}% (Actual FAR: {far_new*100:.2f}%)")
print(f"  Threshold: {th_new:.4f}")

tar_old, far_old, th_old = calculate_tar_at_far(file_old, 0.01)
print(f"\nOld Model (margin 0.2):")
print(f"  TAR @ 1% FAR: {tar_old*100:.2f}% (Actual FAR: {far_old*100:.2f}%)")
print(f"  Threshold: {th_old:.4f}")

print("="*50)
