import numpy as np
import pickle
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve

# Path to the data (latest margin-var embeddings)
pkl_file = "roc_data/embeddings_Gait3D_DeepGaitV2_triplet_margin-var.pkl"

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

# Flatten and Reshape
if len(probe_features.shape) > 2:
    probe_features = probe_features.reshape(probe_features.shape[0], -1)
if len(gallery_features.shape) > 2:
    gallery_features = gallery_features.reshape(gallery_features.shape[0], -1)

# Define splits (90/10)
unique_test_ids = np.unique(np.concatenate([probe_lbls, gallery_lbls]))
num_intruders = int(len(unique_test_ids) * 0.10)
known_ids = unique_test_ids[:-num_intruders]
is_known_probe = np.isin(probe_lbls, known_ids)

gallery_known_mask = np.isin(gallery_lbls, known_ids)
gallery_known_features = gallery_features[gallery_known_mask]

# PCA Reduction to 128 dimensions for stability and alignment with previous tests
print("Applying PCA reduction to 128 dimensions...")
pca = PCA(n_components=128, random_state=42)
gallery_known_pca = pca.fit_transform(gallery_known_features)
probe_pca = pca.transform(probe_features)

print("\n" + "="*50)
print("RUNNING ISOLATION FOREST TESTS")
print("="*50)

results_summary = []

# Testing different n_estimators (number of isolation trees)
for n_est in [100, 200, 500]:
    print(f"Training Isolation Forest with n_estimators={n_est}...")
    # 'auto' contamination allows us to define the threshold later via EER
    clf = IsolationForest(n_estimators=n_est, contamination='auto', random_state=42, n_jobs=-1)
    clf.fit(gallery_known_pca)
    
    # decision_function: higher is "less anomalous" (more likely to be known)
    scores = clf.decision_function(probe_pca)
    
    # Evaluation: EER
    y_true = np.zeros(len(scores))
    y_true[is_known_probe] = 1 # 1 = Known, 0 = Intruder
    
    fpr, tpr, _ = roc_curve(y_true, scores)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    results_summary.append((n_est, eer))
    print(f"EER (n_estimators={n_est}): {eer*100:.2f}%")

print("\n" + "="*50)
print("ISOLATION FOREST RESULTS SUMMARY")
print("="*50)
for n_est, eer in results_summary:
    print(f"IsoForest n_estimators={n_est}: EER = {eer*100:.2f}%")
print("="*50)
