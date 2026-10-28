import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import os

# Nombres descriptivos para el experimento
MODEL_NAME = "DeepGaitV2"
DATASET_NAME = "Gait3D"

pkl_file = "roc_data/roc_curve_data_gait3d.pkl"

if not os.path.exists(pkl_file):
    print(f"No se encontró el archivo {pkl_file}.")
    exit(1)

print(f"Cargando los datos desde {pkl_file}...")
with open(pkl_file, "rb") as f:
    data = pickle.load(f)

dists_genuinas = data['MinDists_Known_Same']
dists_impostoras = data['MinDists_Intruder']

total_genuinas = len(dists_genuinas)
total_impostoras = len(dists_impostoras)

# Generar un barrido (sweep) de umbrales sobre el rango de distancias
all_dists = np.concatenate([dists_genuinas, dists_impostoras])
thresholds = np.linspace(all_dists.min() * 0.9, all_dists.max() * 1.1, 1000)

fars = []
frrs = []

eer_diff = float('inf')
eer_threshold = 0
eer_val = 0

for tau in thresholds:
    # FRR: Distancia genuina > tau (falsamente rechazado)
    frr = np.sum(dists_genuinas > tau) / total_genuinas
    
    # FAR: Distancia impostora <= tau (falsamente aceptado)
    far = np.sum(dists_impostoras <= tau) / total_impostoras
    
    fars.append(far)
    frrs.append(frr)
    
    # Buscar el punto de corte para el EER
    diff = abs(far - frr)
    if diff < eer_diff:
        eer_diff = diff
        eer_threshold = tau
        # Para ser precisos en la gráfica cruzada, tomamos la media de ambos en el cruce
        eer_val = (far + frr) / 2.0

# Convertir a arrays para graficar
fars = np.array(fars) * 100  # a porcentaje
frrs = np.array(frrs) * 100

plt.figure(figsize=(10, 6))

plt.plot(thresholds, frrs, color='blue', lw=2, label='False Rejection Rate (FRR)')
plt.plot(thresholds, fars, color='red', lw=2, label='False Acceptance Rate (FAR)')

# Dibujar punto EER
plt.scatter([eer_threshold], [eer_val * 100], color='black', marker='X', s=100, zorder=5, 
            label=f'Equal Error Rate (EER: {eer_val*100:.2f}% @ dist {eer_threshold:.2f})')
plt.axvline(x=eer_threshold, color='grey', linestyle='--', alpha=0.5)

plt.xlabel('Umbral de Distancia Euclidiana (\u03C4)')
plt.ylabel('Tasa de Error (%)')
plt.title(f'Análisis de Verificación: FAR y FRR vs Umbral\nModelo: {MODEL_NAME} | Dataset: {DATASET_NAME}')
plt.legend(loc="upper right")
plt.grid(True)
plt.xlim(all_dists.min(), all_dists.max())

# Guardar la gráfica con nombre identificativo
output_dir = "roc_data"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f"plot_far_frr_{MODEL_NAME.lower()}_{DATASET_NAME.lower()}.png")

plt.savefig(output_path, bbox_inches='tight')
plt.close()

print(f"Gráfica FAR/FRR generada y guardada en {output_path}")
print(f"EER estimado: {eer_val*100:.2f}%")
print(f"Umbral (Distancia) del EER: {eer_threshold:.4f}")
