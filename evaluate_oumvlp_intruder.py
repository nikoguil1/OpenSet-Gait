import numpy as np
import pickle
from tqdm import tqdm

import torch

def part_based_euc_dist(x, y):
    """
    Computes Euclidean distance for part-based representations using GPU.
    Returns: (N, M) distance matrix.
    """
    x = torch.tensor(x).cuda()
    y = torch.tensor(y).cuda()
    N, C, P = x.shape
    M = y.shape[0]
    dist = torch.zeros((N, M)).cuda()
    for i in range(P):
        _x = x[:, :, i]
        _y = y[:, :, i]
        _x2 = torch.sum(_x**2, dim=1, keepdim=True)
        _y2 = torch.sum(_y**2, dim=1, keepdim=False)
        _dist = _x2 + _y2 - 2 * torch.matmul(_x, _y.t())
        dist += torch.sqrt(torch.clamp(_dist, min=0.0))
    dist = dist / P
    return dist.cpu().numpy()

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
    
    sorted_impostors = np.sort(intruder_dists)
    idx = int(0.01 * len(sorted_impostors))
    if idx >= len(sorted_impostors): idx = len(sorted_impostors) - 1
    
    tar_threshold = sorted_impostors[idx]
    tar_1far = np.mean(known_dists < tar_threshold)
    actual_far = np.mean(intruder_dists < tar_threshold)

    return eer_value, eer_threshold, tar_1far, actual_far, tar_threshold

if __name__ == '__main__':
    pkl_path = "roc_data/oumvlp_all_embeddings.pkl"
    print(f"Loading data from {pkl_path}...")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    features = data['features']
    lbls = np.asarray(data['lbls'])
    types = np.asarray(data['types'])
    views = np.asarray(data['views'])

    print(f"Total features loaded: {features.shape}")

    unique_test_ids = np.unique(lbls)
    num_intruders = int(len(unique_test_ids) * 0.10)
    known_ids = unique_test_ids[:-num_intruders]

    probe_mask = (types == '00')
    gallery_mask = (types == '01')

    probe_features = features[probe_mask]
    probe_lbls = lbls[probe_mask]

    gallery_all_features = features[gallery_mask]
    gallery_all_lbls = lbls[gallery_mask]

    gallery_known_mask = np.isin(gallery_all_lbls, known_ids)
    gallery_known_features = gallery_all_features[gallery_known_mask]
    
    print(f"Total Test IDs: {len(unique_test_ids)}, Known IDs: {len(known_ids)}, Intruders: {num_intruders}")
    print(f"Probe samples: {probe_features.shape[0]}")
    print(f"Gallery 'known' samples: {gallery_known_features.shape[0]}")

    print("Calculating Minimum Distances Chunk-by-Chunk (Nearest Neighbor)...")
    N = probe_features.shape[0]
    min_distances = np.zeros(N)
    chunk_size = 500

    for i in tqdm(range(0, N, chunk_size)):
        end_i = min(i + chunk_size, N)
        probe_chunk = probe_features[i:end_i]
        
        # Calculate full distance matrix for the chunk
        dist_chunk = part_based_euc_dist(probe_chunk, gallery_known_features)
        
        # We only need the minimum distance for intruder detection logic!
        min_distances[i:end_i] = np.min(dist_chunk, axis=1)

    print("Calculating Metrics...")
    eer, eer_th, tar_1far, far, tar_th = calculate_metrics(min_distances, probe_lbls, known_ids)

    print("="*50)
    print("INTRUDER DETECTION METRICS (1-TO-N NEAREST NEIGHBOR)")
    print("="*50)
    print(f"Dataset: OUMVLP | Model: DeepGaitV2")
    print(f"  EER          : {eer*100:.2f}% (Threshold: {eer_th:.4f})")
    print(f"  TAR @ 1% FAR : {tar_1far*100:.2f}% (Actual FAR: {far*100:.2f}%, Threshold: {tar_th:.4f})")
    print("="*50)
