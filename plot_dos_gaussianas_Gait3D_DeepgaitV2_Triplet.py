import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import os
from scipy.stats import gaussian_kde

pkl_file = "roc_data/roc_curve_data_Gait3D_DeepgaitV2_Triplet.pkl"

if not os.path.exists(pkl_file):
    print(f"No se encontró el archivo {pkl_file}.")
    exit(1)

print(f"Cargando los datos desde {pkl_file}...")
with open(pkl_file, "rb") as f:
    data = pickle.load(f)

min_dists_gallery = data.get('MinDists_Known_Same')
min_dists_intruder = data.get('MinDists_Intruder')

if min_dists_gallery is None or min_dists_intruder is None:
    print("Faltan los datos de distancias mínimas en el archivo pkl.")
    exit(1)

min_dists_gallery = min_dists_gallery[min_dists_gallery < np.inf]
min_dists_intruder = min_dists_intruder[min_dists_intruder < np.inf]

plt.figure(figsize=(10, 6))

density_gallery = gaussian_kde(min_dists_gallery)
density_intruder = gaussian_kde(min_dists_intruder)

# Use bounds
bound_min = min(min_dists_gallery.min(), min_dists_intruder.min())
bound_max = max(min_dists_gallery.max(), min_dists_intruder.max())
x_vals = np.linspace(bound_min, bound_max, 500)

med_gallery = np.median(min_dists_gallery)
std_gallery = np.std(min_dists_gallery)
med_intruder = np.median(min_dists_intruder)
std_intruder = np.std(min_dists_intruder)
n_gallery = len(min_dists_gallery)
n_intruder = len(min_dists_intruder)

# Escalar densidades evaluadas por el número total de muestras de cada clase
y_gallery = density_gallery(x_vals) * n_gallery
y_intruder = density_intruder(x_vals) * n_intruder

label_gallery = f'Identidades Conocidas (N={n_gallery}, Med: {med_gallery:.2f}, Std: {std_gallery:.2f})'
label_intruder = f'Intrusos Reales (N={n_intruder}, Med: {med_intruder:.2f}, Std: {std_intruder:.2f})'

plt.plot(x_vals, y_gallery, color='blue', lw=2, label=label_gallery)
plt.fill_between(x_vals, y_gallery, alpha=0.3, color='blue')

plt.plot(x_vals, y_intruder, color='red', lw=2, label=label_intruder)
plt.fill_between(x_vals, y_intruder, alpha=0.3, color='red')

plt.title('Distribución de Distancias a la Galería (Escalado por Muestras) - Gait3D DeepGaitV2 Triplet')
plt.xlabel('Distancia Euclidiana Mínima')
plt.ylabel('Frecuencia Estimada (Nº de Muestras)')
plt.legend()
plt.grid(True)

output_path = "roc_data/dos_gaussianas_verificacion_Gait3D_DeepgaitV2_Triplet.png"
plt.savefig(output_path, bbox_inches='tight')
plt.close()

print(f"Gráfica generada y guardada en {output_path}")
