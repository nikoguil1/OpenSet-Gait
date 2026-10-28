import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_roc_curve(fpr, tpr, eer, save_path):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (EER = {eer*100:.2f}%)')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"ROC graph saved to {save_path}")

def plot_distance_distributions(min_dists_known, min_dists_intruder, min_dists_gallery, save_path):
    plt.figure(figsize=(10, 6))
    
    plt.hist(min_dists_known, bins=50, density=True, alpha=0.5, label="Known Probe vs Gallery (Match)")
    plt.hist(min_dists_intruder, bins=50, density=True, alpha=0.5, label="Intruder Probe vs Gallery (Mismatch)")
    plt.hist(min_dists_gallery, bins=50, density=True, alpha=0.5, label="Gallery vs Gallery (Mismatch)")
    
    plt.xlabel('Euclidean Distance')
    plt.ylabel('Density')
    plt.title('Euclidean Distance Distributions (Open-Set Verification)')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Distributions graph saved to {save_path}")

def plot_tar_far_thresholds(fpr, tpr, thresholds, save_path):
    # La salida de la curva ROC empieza con un threshold infinito por convención.
    # Además, habíamos negativizado la distancia (-dist) para el Score de sklearn.
    # Revertiremos esto de vuelta a valores de distancia Euclidianos reales purgados del infinito.
    valid_idx = thresholds < np.inf
    fpr = fpr[valid_idx]
    tpr = tpr[valid_idx]
    dist_thresholds = -thresholds[valid_idx]

    plt.figure(figsize=(10, 6))
    plt.plot(dist_thresholds, tpr, color='green', lw=2, label='True Acceptance Rate (TAR / Recall)')
    plt.plot(dist_thresholds, fpr, color='red', lw=2, label='False Acceptance Rate (FAR / Intrusion)')
    
    plt.xlabel('Distance Threshold')
    plt.ylabel('Rate')
    plt.title('TAR and FAR vs. Distance Threshold')
    plt.legend(loc="center right")
    plt.grid(True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"TAR/FAR threshold graph saved to {save_path}")

def plot_frr_far_thresholds(fpr, tpr, thresholds, eer, save_path):
    valid_idx = thresholds < np.inf
    fpr = fpr[valid_idx]
    frr = 1 - tpr[valid_idx] # False Rejection Rate = 1 - TPR
    dist_thresholds = -thresholds[valid_idx]

    plt.figure(figsize=(10, 6))
    plt.plot(dist_thresholds, frr, color='blue', lw=2, label='False Rejection Rate (FRR)')
    plt.plot(dist_thresholds, fpr, color='red', lw=2, label='False Acceptance Rate (FAR)')
    
    # Dibujar un marcador donde se crucen (donde son aproximadamente iguales, i.e., el EER)
    if eer is not None:
        idx_eer = np.nanargmin(np.absolute((fpr - frr)))
        eer_threshold = dist_thresholds[idx_eer]
        plt.scatter(eer_threshold, eer, color='black', marker='x', s=100, zorder=5, label=f'EER Crossing ({eer*100:.2f}%)')
    
    plt.xlabel('Distance Threshold')
    plt.ylabel('Rate')
    plt.title('FRR and FAR vs. Distance Threshold')
    plt.legend(loc="center right")
    plt.grid(True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"FRR/FAR threshold graph saved to {save_path}")

if __name__ == "__main__":
    pkl_file = "roc_data/roc_curve_data_gait3d.pkl"
    if not os.path.exists(pkl_file):
        print(f"File {pkl_file} not found. Make sure the evaluation script finished successfully.")
        import sys
        sys.exit(1)
        
    print(f"Loading {pkl_file}...")
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)
        
    if not os.path.exists("roc_data"):
        os.makedirs("roc_data")
        
    plot_roc_curve(
        data['FPR'], data['TPR'], data['EER'], 
        "roc_data/eer_roc_graph.png"
    )
    
    # Check if distributions are present
    if 'MinDists_Known' in data and 'MinDists_Intruder' in data and 'MinDists_Gallery' in data:
        plot_distance_distributions(
            data['MinDists_Known'], data['MinDists_Intruder'], data['MinDists_Gallery'],
            "roc_data/distance_distributions_graph.png"
        )
    else:
        print("Distance distributions data not found in pickle file.")
        
    plot_tar_far_thresholds(
        data['FPR'], data['TPR'], data['Thresholds'],
        "roc_data/far_tar_threshold_graph.png"
    )
    
    plot_frr_far_thresholds(
        data['FPR'], data['TPR'], data['Thresholds'], data.get('EER', None),
        "roc_data/far_frr_threshold_graph.png"
    )
