import numpy as np
import pickle
import os

pkl_file = "roc_data/roc_curve_data_gait3d.pkl"

if not os.path.exists(pkl_file):
    print(f"No se encontró el archivo {pkl_file}.")
    exit(1)

with open(pkl_file, "rb") as f:
    data = pickle.load(f)

dists_genuinas = data['MinDists_Known_Same']
dists_impostoras = data['MinDists_Intruder']

total_genuinas = len(dists_genuinas)
total_impostoras = len(dists_impostoras)

# Combine and sort all distances to use as candidate thresholds
all_dists = np.concatenate([dists_genuinas, dists_impostoras])
thresholds = np.sort(all_dists)

eer_diff = float('inf')
eer_threshold = 0
eer_far = 0
eer_frr = 0
eer_falsos_rechazos = 0
eer_falsas_aceptaciones = 0

print("Calculando umbrales, espere por favor...")

for tau in thresholds:
    # FRR: Genuine distance > tau (falsely rejected)
    falsos_rechazos = np.sum(dists_genuinas > tau)
    frr = falsos_rechazos / total_genuinas
    
    # FAR: Impostor distance <= tau (falsely accepted)
    falsas_aceptaciones = np.sum(dists_impostoras <= tau)
    far = falsas_aceptaciones / total_impostoras
    
    # EER is where the difference between FAR and FRR is minimized
    diff = abs(far - frr)
    if diff < eer_diff:
        eer_diff = diff
        eer_threshold = tau
        eer_far = far
        eer_frr = frr
        eer_falsos_rechazos = falsos_rechazos
        eer_falsas_aceptaciones = falsas_aceptaciones

print("\n" + "="*50)
print(f"RESULTADOS EER PARA GAIT3D")
print("="*50)
print(f"Umbral Óptimo (Distancia EER): {eer_threshold:.4f}")
print(f"Tasa FAR (Falsos Accesos):     {eer_far * 100:.2f}%")
print(f"Tasa FRR (Falsos Rechazos):    {eer_frr * 100:.2f}%")
print("-" * 50)
print("Desglose de conteos en el EER:")
print(f"Total de Genuinos Evaluados:    {total_genuinas}")
print(f"Total de Impostores Evaluados:  {total_impostoras}")
print(f"Falsos Rechazos (Usuarios bloqueados): {eer_falsos_rechazos}")
print(f"Falsas Aceptaciones (Intrusos colados): {eer_falsas_aceptaciones}")
print("="*50)
