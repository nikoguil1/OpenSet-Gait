# Arquitectura — OpenSet-Gait

Base: fork de [OpenGait](https://github.com/ShiqiYu/OpenGait) (upstream `f754f6f`). Modelo de referencia: **DeepGaitV2 (modo `p3d`)** sobre siluetas, configuración `configs/deepgaitv2/DeepGaitV2_gait3d.yaml`. Evaluación open-set encima de los embeddings (ver `opengait/evaluation/evaluator.py`).

## 1. Diagrama

```mermaid
flowchart TD
    A["Siluetas pkl<br/>[n, s, 64, 44]"] --> T["Transform<br/>BaseSilCutting + aug. (train)"]
    T --> L0["layer0: conv3x3 + BN + ReLU (2D por frame)"]
    L0 --> L1["layer1: 1× BasicBlock2D, 64 ch"]
    L1 --> L2["layer2: 4× BasicBlockP3D, 128 ch, stride 2"]
    L2 --> L3["layer3: 4× BasicBlockP3D, 256 ch, stride 2"]
    L3 --> L4["layer4: 1× BasicBlockP3D, 512 ch"]
    L4 --> TP["TP: max temporal (PackSequenceWrapper)"]
    TP --> HPP["HPP: 16 franjas horizontales"]
    HPP --> FC["SeparateFCs 512→256"]
    FC -->|embed_1| BN["SeparateBNNecks"]
    BN -->|embed_2| E{"use_emb2?"}
    FC -->|embed_1| E
    E -->|embed| ML["Pérdida métrica<br/>Triplet / BHCn / SupCon"]
    BN -->|logits| CE["CrossEntropy (scale 16)"]
    E -->|embed| INF["Inferencia: embeddings [n, 256, 16]"]
    INF --> EV["Evaluador: distancia euclídea por partes<br/>cerrado (R1, mAP) + verificación (EER, TAR@FAR)<br/>+ intrusos open-set"]
```

## 2. Dimensiones (Gait3D, entrenamiento)

Batch del `TripletSampler`: 32 identidades × 4 secuencias → `n = 128`; `s = 30` frames.

| Etapa | Shape | Notas |
|---|---|---|
| Entrada | `[n, 1, s, 64, 44]` | `assert` anchura ∈ {44, 88} |
| layer0 / layer1 | `[n, 64, s, 64, 44]` | |
| layer2 | `[n, 128, s, 32, 22]` | stride espacial 2 |
| layer3 | `[n, 256, s, 16, 11]` | stride espacial 2 |
| layer4 | `[n, 512, s, 16, 11]` | |
| TP (max sobre `s`) | `[n, 512, 16, 11]` | respeta `seqL` |
| HPP (16 bins) | `[n, 512, 16]` | `p = 16` partes |
| `embed_1` (SeparateFCs) | `[n, 256, 16]` | |
| `embed_2`, `logits` (BNNecks) | `[n, 256, 16]`, `[n, 3000, 16]` | 3000 IDs de entrenamiento |

`embed = embed_2 if use_emb2 else embed_1` se usa tanto en la pérdida métrica como en la inferencia. Upstream entrenaba la triplet siempre con `embed_1`; en este fork solo cambia con `use_emb2: true` (hoy únicamente `DeepGaitV2_Gait3D_SupCons_Euc.yaml`).

## 3. Funciones de pérdida

Todas operan por parte (`[p, n, c]`) y se suman a la CE del BNNeck (peso 1.0).

| Tipo (`loss_cfg.type`) | Fichero | Definición |
|---|---|---|
| `TripletLoss` | `losses/triplet.py` | `relu(d_ap − d_an + m)`, `m = 0.2`, media de no-ceros. Registra media/std de `d_ap` y `d_an`. |
| `BHCLoss` | `losses/bhc.py` | Batch-hard: `relu(max d_ap − m1) + relu(m2 − min d_an)`; `m1 = 0.2`, `m2 = 0.5` y se duplica cada 20k iteraciones. |
| `BHCLossV2` | `losses/bhcV2.py` | `relu(d_p) + relu(m − d_n)` con margen fijo `m` (config: 10). |
| `BHCLossV3` | `losses/bhcV3.py` | Grouped BHCn (ec. 6 de Aljazaerly et al. 2023): pérdidas promediadas entre partes, después hardest positivo/negativo y al cuadrado; `m = 8`, +2 cada 15k iteraciones. |
| `SupConLoss_Lp` | `losses/supconloss.py` | SupCon con distancia euclídea sobre embeddings normalizados L2, `T = 0.07`. |
| `SupConLoss_Euc` | `losses/supconloss.py` | SupCon con distancia euclídea **sin normalizar**, `T = 1`. Distancia calculada a mano (evita el NaN del gradiente de `torch.cdist` en 0). |
| `CrossEntropyLoss` | `losses/ce.py` | Sobre `logits` del BNNeck, `scale = 16`. |

## 4. Evaluación open-set (`opengait/evaluation/openset.py`)

```mermaid
flowchart LR
    E["embeddings [N, 256, 16]"] --> D["cuda_dist: probe × galería COMPLETA<br/>(una sola vez)"]
    D --> C["Cerrado: R1/5/10, mAP, mINP"]
    D --> S["Por partición: columnas de IDs<br/>desconocidos eliminadas"]
    S --> M["R1 conocidos · EER, TAR@FAR (pares)<br/>DIR/FNIR@FPIR, OSCR, AUROC (d_min)"]
    M --> A["media ± std por ratio → log, TensorBoard<br/>roc_data/{dataset}_{save_name}_openset.pkl"]
```

- **Particiones**: para cada `unknown_ratio` (por defecto 10/30/50 %), `n_splits` (10) conjuntos aleatorios de IDs desconocidos, con semilla fija, tomados de las identidades del probe. Sus secuencias salen de la galería; todos los probes se conservan. Un probe es *conocido* si y solo si su ID sigue en la galería reducida. Además se calcula `legacy10` (último 10 % de IDs ordenados), la partición del código anterior, para comparar con resultados previos.
- **Regla de decisión**: el probe se acepta si `d_min < τ`, siendo `d_min` la distancia al vecino más cercano de la galería; si se acepta, se identifica con ese vecino. DIR@FPIR=x fija τ para que, como mucho, una fracción x de los probes desconocidos se acepte.
- **Configuración** (opcional, en `evaluator_cfg`):
  ```yaml
  openset_cfg:
    unknown_ratios: [0.1, 0.3, 0.5]
    n_splits: 10
    seed: 0
    fpir: [0.01, 0.1]
    far: [0.01, 0.001]
    legacy_split: true
    output_dir: roc_data
  ```
- **Salida pkl**: `results[split]` contiene `unknown_ids`, `metrics` y `curves` por partición (curva DIR–FPIR, `d_min`, `is_known`, `correct`, ROC de verificación submuestreada) y `summary` (media, std). También guarda `probe_features`, `gallery_features`, `probe_lbls` y `gallery_lbls` (galería completa), con las mismas claves que los pkl antiguos.
- Integrado en `evaluate_Gait3D`. SUSTech1K, CCPG y CCGR se conectan al preparar los datos (CU-123nktjx7ym): basta con pasar `dist`, `probe_lbls` y `gallery_lbls` a `run_protocol`.
- Tests: `python tests/test_openset.py` (embeddings perfectos, aleatorios, DIR@FPIR calculado a mano, ausencia de fugas, reproducibilidad de particiones).
