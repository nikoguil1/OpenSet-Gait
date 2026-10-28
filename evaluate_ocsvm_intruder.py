import numpy as np
import pickle
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve

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

# Flatten embeddings if they have more than 2 dimensions
if len(probe_features.shape) > 2:
    probe_features = probe_features.reshape(probe_features.shape[0], -1)
if len(gallery_features.shape) > 2:
    gallery_features = gallery_features.reshape(gallery_features.shape[0], -1)

# Define the set of known vs unknown based on the same logic as before
unique_test_ids = np.unique(np.concatenate([probe_lbls, gallery_lbls]))
num_intruders = int(len(unique_test_ids) * 0.10) # 10% intruders
known_ids = unique_test_ids[:-num_intruders]
is_known_probe = np.isin(probe_lbls, known_ids)

gallery_known_mask = np.isin(gallery_lbls, known_ids)
gallery_known_features = gallery_features[gallery_known_mask]

# PCA Reduction (recommended for OCSVM performance and stability in high dimensions)
print("Applying PCA reduction to 128 dimensions...")
pca = PCA(n_components=128, random_state=42)
gallery_known_pca = pca.fit_transform(gallery_known_features)
probe_pca = pca.transform(probe_features)

print("\n" + "="*50)
print("RUNNING ONE-CLASS SVM TESTS")
print("="*50)

results_summary = []

# Testing different nu values (nu is approximately the ratio of training outliers)
for nu_val in [0.01, 0.05, 0.1, 0.2]:
    print(f"Training One-Class SVM with nu={nu_val}...")
    clf = OneClassSVM(kernel='rbf', gamma='scale', nu=nu_val)
    clf.fit(gallery_known_pca)
    
    # decision_function gives signed distance to the separating hyperplane
    # Higher values are more "inside" the target class
    scores = clf.decision_function(probe_pca)
    
    # Evaluation: EER
    y_true = np.zeros(len(scores))
    y_true[is_known_probe] = 1 # 1 = Known, 0 = Intruder
    
    fpr, tpr, _ = roc_curve(y_true, scores)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    results_summary.append((nu_val, eer))
    print(f"EER (nu={nu_val}): {eer*100:.2f}%")

print("\n" + "="*50)
print("OCSVM RESULTS SUMMARY")
print("="*50)
for nu_val, eer in results_summary:
    print(f"OCSVM nu={nu_val}: EER = {eer*100:.2f}%")
print("="*50)
