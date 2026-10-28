import numpy as np
import pickle

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

def calculate_metrics(dist_matrix, probe_lbls, known_ids):
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
    
    # EER
    eer_value = (fars[eer_idx] + frrs[eer_idx]) / 2
    eer_threshold = thresholds[eer_idx]
    
    # TAR at 1% FAR
    # A sample is accepted if dist < threshold
    sorted_impostors = np.sort(intruder_dists)
    idx = int(0.01 * len(sorted_impostors))
    if idx >= len(sorted_impostors): idx = len(sorted_impostors) - 1
    tar_threshold = sorted_impostors[idx]
    
    tar_1far = np.mean(known_dists < tar_threshold)
    actual_far = np.mean(intruder_dists < tar_threshold)

    return eer_value, eer_threshold, tar_1far, tar_threshold

# Load CASIA-B data
pkl_file = "roc_data/casiab_all_embeddings.pkl"
with open(pkl_file, "rb") as f:
    data = pickle.load(f)

feature = data['features']
label = data['lbls']
seq_type = data['types']

unique_test_ids = np.unique(label)
num_intruders = 5 # 10%
known_ids = unique_test_ids[:-num_intruders]

gallery_mask_nm = np.isin(seq_type, ['nm-01', 'nm-02', 'nm-03', 'nm-04'])
gallery_known_mask = gallery_mask_nm & np.isin(label, known_ids)
gallery_features = feature[gallery_known_mask, :]

conditions = {
    'NM': ['nm-05', 'nm-06'],
    'BG': ['bg-01', 'bg-02'],
    'CL': ['cl-01', 'cl-02']
}

for cond, seqs in conditions.items():
    probe_mask = np.isin(seq_type, seqs)
    probe_features = feature[probe_mask, :]
    probe_lbls = label[probe_mask]
    
    dist_matrix = part_based_euc_dist(probe_features, gallery_features)
    eer, eer_th, tar_1far, tar_th = calculate_metrics(dist_matrix, probe_lbls, known_ids)
    print(f"Condition: {cond}")
    print(f"  EER = {eer*100:.2f}% (Threshold: {eer_th:.4f})")
    print(f"  TAR@1%FAR = {tar_1far*100:.2f}% (Threshold: {tar_th:.4f})\n")
