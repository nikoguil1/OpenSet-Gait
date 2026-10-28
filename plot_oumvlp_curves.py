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
    pkl_path = "roc_data/oumvlp_all_embeddings.pkl"
    print(f"Loading data from {pkl_path}...")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    features = data['features']
    lbls = np.asarray(data['lbls'])
    types = np.asarray(data['types'])
    # views = np.asarray(data['views']) # Not used strictly for calculating dists

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
        
        # Absolute minimum (for FAR/FRR matching)
        min_distances_all[i:end_i] = np.min(dist_chunk, axis=1)
        
        # Split into same/diff for gaussian plotting
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
        fars.append(np.mean(intruder_dists <= t))  # False Acceptance
        frrs.append(np.mean(known_dists > t))      # False Rejection

    fars = np.array(fars) * 100
    frrs = np.array(frrs) * 100

    diff = np.abs(fars - frrs)
    eer_idx = np.argmin(diff)
    eer = (fars[eer_idx] + frrs[eer_idx]) / 2

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, fars, label='FAR (Intrusos Aceptados)', color='red', linewidth=2)
    plt.plot(thresholds, frrs, label='FRR (Conocidos Rechazados)', color='blue', linewidth=2)
    plt.axvline(x=thresholds[eer_idx], color='black', linestyle='--', label=f'EER = {eer:.2f}%')
    plt.title('OUMVLP - Open-Set Intruder Detection (FAR vs FRR)', fontsize=14)
    plt.xlabel('Umbral de Distancia (Threshold)', fontsize=12)
    plt.ylabel('Tasa de Error (%)', fontsize=12)
    
    # We use log scale if values are very small, but linear is clearer for zero-crossing
    # plt.yscale('log') 
    
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('oumvlp_far_frr_curve.png', dpi=300)

    print("Generating Gaussian Distributions (Densities)...")
    plt.figure(figsize=(10, 6))
    sns.kdeplot(min_dists_known_same, fill=True, color="blue", label="Genuinos (vs Su ID)")
    sns.kdeplot(min_dists_known_diff, fill=True, color="green", label="Genuinos (vs OTRA ID)")
    sns.kdeplot(min_dists_intruder, fill=True, color="red", label="Intrusos (vs Cualquier ID)")

    plt.title('OUMVLP Distribución de Distancias a la Galería (1-a-N)', fontsize=14)
    plt.xlabel('Distancia Mínima', fontsize=12)
    plt.ylabel('Densidad', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig('oumvlp_distance_gaussians.png', dpi=300)

    print("Successfully saved 'oumvlp_far_frr_curve.png' and 'oumvlp_distance_gaussians.png'.")
