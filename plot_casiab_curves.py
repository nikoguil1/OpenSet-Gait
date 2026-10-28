import numpy as np
import pickle
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import seaborn as sns

def part_based_euc_dist(x, y):
    """Computes Euclidean distance for part-based representations using GPU."""
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

if __name__ == '__main__':
    pkl_path = "roc_data/casiab_all_embeddings.pkl"
    print(f"Loading ALL CASIA-B data from {pkl_path}...")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    features = data['features']
    lbls = np.asarray(data['lbls'])
    types = np.asarray(data['types'])

    # CASIA-B Definition
    nm_probe = ['nm-05', 'nm-06']
    bg_probe = ['bg-01', 'bg-02']
    cl_probe = ['cl-01', 'cl-02']
    gallery_seqs = ['nm-01', 'nm-02', 'nm-03', 'nm-04']
    
    probe_mask = np.isin(types, nm_probe + bg_probe + cl_probe)
    gallery_mask = np.isin(types, gallery_seqs)

    probe_features = features[probe_mask]
    probe_lbls = lbls[probe_mask]
    
    gallery_all_features = features[gallery_mask]
    gallery_all_lbls = lbls[gallery_mask]

    unique_test_ids = np.unique(np.concatenate([probe_lbls, gallery_all_lbls]))
    num_intruders = int(len(unique_test_ids) * 0.10)
    if num_intruders == 0: num_intruders = 1
    known_ids = unique_test_ids[:-num_intruders]

    gallery_known_mask = np.isin(gallery_all_lbls, known_ids)
    gallery_known_features = gallery_all_features[gallery_known_mask]
    gallery_known_lbls = gallery_all_lbls[gallery_known_mask]

    N = probe_features.shape[0]
    chunk_size = 500

    min_dists_known_same = []
    min_dists_known_diff = []
    min_dists_intruder = []
    min_distances_all = np.zeros(N)

    print("Calculating Distances & Separating Same/Diff Classes Chunk-by-Chunk...")
    for i in tqdm(range(0, N, chunk_size)):
        end_i = min(i + chunk_size, N)
        probe_chunk = probe_features[i:end_i]
        probe_lbls_chunk = probe_lbls[i:end_i]
        
        dist_chunk = part_based_euc_dist(probe_chunk, gallery_known_features)
        
        min_distances_all[i:end_i] = np.min(dist_chunk, axis=1)
        
        for j, p_lbl in enumerate(probe_lbls_chunk):
            dists = dist_chunk[j]
            if p_lbl in known_ids:
                same_mask = (gallery_known_lbls == p_lbl)
                diff_mask = ~same_mask
                if np.any(same_mask):
                    min_dists_known_same.append(np.min(dists[same_mask]))
                if np.any(diff_mask):
                    min_dists_known_diff.append(np.min(dists[diff_mask]))
            else:
                if len(dists) > 0:
                    min_dists_intruder.append(np.min(dists))

    min_dists_known_same = np.array(min_dists_known_same)
    min_dists_known_diff = np.array(min_dists_known_diff)
    min_dists_intruder = np.array(min_dists_intruder)

    print("Generating FAR / FRR Curve...")
    is_known_mask = np.isin(probe_lbls, known_ids)
    known_dists = min_distances_all[is_known_mask]
    intruder_dists = min_distances_all[~is_known_mask]

    thresholds = np.linspace(min_distances_all.min(), min_distances_all.max(), 1000)
    fars = []
    frrs = []

    for t in thresholds:
        fars.append(np.mean(intruder_dists <= t))  
        frrs.append(np.mean(known_dists > t))      

    fars = np.array(fars)
    frrs = np.array(frrs)
    diff = np.abs(fars - frrs)
    eer_idx = np.argmin(diff)
    eer = (fars[eer_idx] + frrs[eer_idx]) / 2

    sorted_impostors = np.sort(intruder_dists)
    idx = int(0.01 * len(sorted_impostors))
    if idx >= len(sorted_impostors): idx = len(sorted_impostors) - 1
    tar_threshold = sorted_impostors[idx]
    tar_1far = np.mean(known_dists < tar_threshold)
    actual_far = np.mean(intruder_dists < tar_threshold)

    print("="*50)
    print("INTRUDER DETECTION METRICS (ALL CASIA-B SCENARIOS COMBINED)")
    print("="*50)
    print(f"Dataset: CASIA-B (NM+BG+CL) | Model: DeepGaitV2")
    print(f"  EER          : {eer*100:.2f}% (Threshold: {thresholds[eer_idx]:.4f})")
    print(f"  TAR @ 1% FAR : {tar_1far*100:.2f}% (Actual FAR: {actual_far*100:.2f}%, Threshold: {tar_threshold:.4f})")
    print("="*50)

    fars = fars * 100
    frrs = frrs * 100


    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, fars, label='FAR (Intrusos Aceptados)', color='red', linewidth=2)
    plt.plot(thresholds, frrs, label='FRR (Conocidos Rechazados)', color='blue', linewidth=2)
    plt.axvline(x=thresholds[eer_idx], color='black', linestyle='--', label=f'EER = {eer:.2f}%')
    plt.title('CASIA-B - Open-Set Intruder Detection (FAR vs FRR)', fontsize=14)
    plt.xlabel('Umbral de Distancia (Threshold)', fontsize=12)
    plt.ylabel('Tasa de Error (%)', fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('casiab_new_far_frr_curve.png', dpi=300)

    print("Generating Gaussian Distributions (Densities)...")
    plt.figure(figsize=(10, 6))
    if len(min_dists_known_same) > 2:
        sns.kdeplot(min_dists_known_same, fill=True, color="blue", label="Genuinos (vs Su ID)")
    if len(min_dists_known_diff) > 2:
        sns.kdeplot(min_dists_known_diff, fill=True, color="green", label="Genuinos (vs OTRA ID)")
    if len(min_dists_intruder) > 2:
        sns.kdeplot(min_dists_intruder, fill=True, color="red", label="Intrusos (vs Cualquier ID)")

    plt.title('CASIA-B Distribución de Distancias a la Galería (1-a-N)', fontsize=14)
    plt.xlabel('Distancia Mínima', fontsize=12)
    plt.ylabel('Densidad', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig('casiab_new_distance_gaussians.png', dpi=300)

    print("Successfully saved CASIA-B plots.")
