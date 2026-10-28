import numpy as np
import pickle
from sklearn.neighbors import LocalOutlierFactor

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

def calculate_metrics(anomaly_scores, probe_lbls, known_ids):
    # Invert anomaly scores so lower distance equivalent means MORE NORMAL.
    # LOF score_samples: more negative = more anomaly.
    # If distance = -score, then larger distance = more anomaly.
    distances = -anomaly_scores
    
    is_known_mask = np.isin(probe_lbls, known_ids)
    known_dists = distances[is_known_mask]
    intruder_dists = distances[~is_known_mask]

    thresholds = np.linspace(distances.min(), distances.max(), 1000)
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

print(f"Calculando Matriz de Distancia Gallery-Gallery para entrenar LOF...")
dist_train = part_based_euc_dist(gallery_known_features, gallery_known_features)

print("Entrenando Local Outlier Factor (k=20)...")
lof = LocalOutlierFactor(n_neighbors=20, metric='precomputed', novelty=True)
lof.fit(dist_train)

print(f"Calculando Matriz de Distancia Probe-Gallery para Testeo LOF...")
dist_test = part_based_euc_dist(probe_features, gallery_known_features)

print("Prediciendo Scores de Anomalía...")
anomaly_scores = lof.score_samples(dist_test)

print("Calculando Métricas...")
eer, eer_th, tar_1far, far, tar_th = calculate_metrics(anomaly_scores, probe_lbls, known_ids)

print("="*50)
print("INTRUDER DETECTION METRICS (LOCAL OUTLIER FACTOR)")
print("="*50)
print(f"Dataset: Gait3D | Model: DeepGaitV2 (margin=var)")
print(f"  EER          : {eer*100:.2f}% (Threshold: {-eer_th:.4f})")
print(f"  TAR @ 1% FAR : {tar_1far*100:.2f}% (Actual FAR: {far*100:.2f}%, Threshold: {-tar_th:.4f})")
print("="*50)
