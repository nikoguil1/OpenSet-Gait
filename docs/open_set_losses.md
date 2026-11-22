# Funciones de pérdida para entrenamiento en conjunto abierto

Referencia de las pérdidas usadas en OpenSet-Gait para entrenar modelos de reconocimiento de la marcha en conjunto abierto (*open-set*): formulación, explicación breve, implementación y parámetros de configuración.

## 1. Notación y pérdida total

- El modelo produce, para cada secuencia $x$, un embedding por parte horizontal $f_k(x) \in \mathbb{R}^c$, con $k = 1..P$ ($P = 16$ en GaitBase/DeepGaitV2, $15$ en SwinGait-3D). Todas las pérdidas se calculan **por parte** y se promedian sobre las partes.
- Distancia: euclídea **sin normalizar** por parte, $d_k(a, b) = \lVert f_k(a) - f_k(b) \rVert_2$. Es la misma que usa el evaluador (`metric: euc`, media de las distancias por parte). Para no recargar la notación se omite el índice $k$.
- Batch: sampler de Su et al. (`TripletNoRepeatSampler`), con 128 secuencias distintas y hasta 4 por sujeto, o batches $P\times K$ de OpenGait.

La pérdida total de todos los experimentos open-set es

$$
\mathcal{L} = \mathcal{L}_{\text{tri}} + \mathcal{L}_{\text{CE}} + \lambda\, \mathcal{L}_{\text{open}},
$$

donde $\mathcal{L}_{\text{open}}$ es `OpenSetLoss` (Su et al.) u `OpenPairLoss` (propia), y $\lambda$ es su `loss_term_weight`: 1 con OpenSetLoss y 0.5 con OpenPair. La triplet y la pérdida open-set actúan sobre `embed_1` (salida de las FC separadas). La CE actúa sobre los logits del BNNeck.

## 2. Simulación open-set dentro del batch

`OpenSetLoss` y `OpenPairLoss` reproducen en cada batch el escenario de evaluación open-set:

1. Los sujetos del batch se reparten en $F = 4$ *folds* (`num_folds`).
2. En cada fold:
   - Los sujetos del fold hacen de **probes no registrados** $\mathcal{P}_{NM}$ (intrusos).
   - Cada uno de los demás sujetos $j$ aporta:
     - una **plantilla de galería** $g_j$, la media de los embeddings de la primera mitad de sus secuencias, $\lceil n_j/2 \rceil$ (al menos 1);
     - **probes registrados** $\mathcal{P}_M$, el resto de sus secuencias.
   - $G = \{g_j\}$ es la galería del fold, de unos 24 sujetos con el batch de Su.
3. La pérdida se acumula sobre los 4 folds.

Cada secuencia es intruso en exactamente un fold. Ninguna secuencia está a la vez en plantilla y en probe (sin fuga). En test, la plantilla es también la media de las secuencias de galería de cada sujeto.

## 3. Triplet (`TripletLoss`, `TripletLossSu`)

$$
\mathcal{L}_{\text{tri}} = \frac{1}{|\mathcal{T}_{+}|} \sum_{(a,p,n)\in\mathcal{T}} \max\big(0,\; d(a,p) - d(a,n) + m\big)
$$

- $\mathcal{T}$ contiene todas las tripletas del batch (*batch-all*), con ancla $a$, positivo $p$ de la misma identidad y negativo $n$ de otra.
- $\mathcal{T}_{+}$ son las tripletas activas (hinge $> 0$). Se promedia solo sobre ellas (`AvgNonZeroReducer`), así la pérdida no se desvanece cuando la mayoría cumple el margen.
- $m = 0.2$.

**Qué hace:** acerca las secuencias de la misma persona y aleja las de personas distintas, en términos relativos a cada ancla. Es la pérdida métrica de conjunto cerrado.

**No resuelve el conjunto abierto:** no impone una escala de distancias común a todas las identidades, y eso es lo que necesita un umbral global de rechazo.

**Implementación:**
- `losses/triplet.py` (batches $P\times K$).
- `losses/triplet_su.py` (port de Su et al.): construye las tripletas ancla a ancla y admite sujetos con distinto número de secuencias. En batches $P\times K$ da la misma pérdida.

Registra `mean_dist`, la distancia media del batch, que sirve para vigilar la escala del espacio.

## 4. Cross-entropy con BNNeck (`CrossEntropyLoss`)

$$
\mathcal{L}_{\text{CE}} = -\frac{1}{N}\sum_{i} \sum_{y} q_{i,y} \log \operatorname{softmax}\big(s \cdot \cos(\hat f(x_i), w_y)\big)_y
$$

- $\hat f$ es el embedding tras BatchNorm y normalización L2 por parte (BNNeck).
- $w_y$ son los pesos normalizados del clasificador, uno por identidad de entrenamiento.
- $s = 16$ (`scale`).
- $q$ es la etiqueta con *label smoothing* $\varepsilon = 0.1$.

**Qué hace:** clasificación entre las identidades de entrenamiento en espacio coseno. Estabiliza el entrenamiento y mejora el conjunto cerrado. El clasificador no se usa en test.

**Implementación:** `losses/ce.py`.

## 5. OpenSetLoss de Su et al. (`OpenSetLoss`)

Su et al., *Open-Set Biometrics: Beyond Good Closed-Set Models*, ECCV 2024, ec. 6–12. Port de `prevso1088/open-set-biometrics` en `losses/open_set.py`.

Las distancias se convierten en puntuaciones $s = 1/(1+d)$, de modo que una puntuación mayor indica más parecido. Con la simulación de la sección 2, para cada probe registrado $x \in \mathcal{P}_M$ de identidad $y$ se define $s_x = s(x, g_y)$.

**Rango suave**, que aproxima la posición del verdadero en el ranking:

$$
\tilde r(x) = \sum_{g \in G} \sigma\big(\gamma\, (s(x,g) - s_x)\big)
$$

La suma incluye la propia plantilla, que aporta $\sigma(0) = 0.5$.

**Identification-Detection Loss (IDL).** Es un DIR suave en el que cada puntuación de un intruso, $\theta \in \Theta = \{ s(u, g) : u \in \mathcal{P}_{NM},\, g\in G\}$, hace de umbral:

$$
\widetilde{\text{DIR}} = \frac{1}{|\Theta|\,|\mathcal{P}_M|}\sum_{\theta\in\Theta}\sum_{x\in\mathcal{P}_M}
\underbrace{\sigma\big(\beta\,(r - \tilde r(x))\big)}_{S_{id}:\ \text{rango} \le r}\;
\underbrace{\sigma\big(\alpha\,(s_x - \theta)\big)}_{S_{det}:\ \text{supera el umbral}}
$$

**Relative Threshold Minimization (RTM).** Es la puntuación de cada intruso contra la galería, ponderada con softmax para dar más peso a las plantillas más cercanas:

$$
\text{RTM} = \frac{1}{|\mathcal{P}_{NM}|}\sum_{u\in\mathcal{P}_{NM}} \sum_{g\in G} \operatorname{softmax}_g\big(-d(u,g)\big)\; s(u,g)
$$

**Pérdida:**

$$
\mathcal{L}_{\text{OSL}} = w_{det}\,\big(1 - \widetilde{\text{DIR}}\big) + \lambda_{RTM}\,\text{RTM}
$$

**Qué hace:** optimiza directamente una versión diferenciable del DIR@FPIR, la métrica de identificación en conjunto abierto. La IDL premia que el genuino quede en el top-$r$ y por encima de los umbrales que marcan los intrusos. La RTM baja las puntuaciones de los intrusos, es decir, el propio umbral.

**Limitación observada:** como $s = 1/(1+d)$ depende de la escala absoluta de $d$, la red puede bajar la pérdida inflando todas las distancias sin separar mejor. Lo motivó la calibración siguiente.

**Parámetros:**

| Símbolo | Parámetro | Gait3D (Su) | Significado |
|---|---|---|---|
| $\alpha$ | `score_scale` | 9 (SwinGait) / 6 | pendiente de la detección |
| $\beta$ | `rank_scale` | 0.2 | pendiente del término de rango |
| $\gamma$ | `rank_diff_scale` | 9 (SwinGait) / 6 | pendiente del rango suave |
| $r$ | `rank` | 1 | rango objetivo |
| $\lambda_{RTM}$ | `rtm_weight` | 4 | peso de la RTM |
| $w_{det}$ | `det_weight` | 1 | peso de $1-\widetilde{\text{DIR}}$ |
| $F$ | `num_folds` | 4 | folds de la simulación |

Registra `dirs`, `nm_scores` (RTM) e `Id` / `Det`, que son las medias de $S_{id}$ y $S_{det}$.

### 5.1 Variante sin identificación (`identification: false`)

$$
\widetilde{\text{DIR}}_{det} = \frac{1}{|\Theta|\,|\mathcal{P}_M|}\sum_{\theta}\sum_{x} \sigma\big(\alpha\,(s_x - \theta)\big)
$$

Elimina $S_{id}$ (el rango suave), de modo que solo queda la detección: que el genuino supere los umbrales de los intrusos. $w_{det}$ (`det_weight`) compensa el peso. Configs `SwinGait3D_Su2024_Gait3D_OpenSetLoss_NoId_w{1,4,10}.yaml`.

### 5.2 IDL calibrada (`scale_invariant: true`, `rank_self_term: false`)

Las puntuaciones se calculan sobre distancias reescaladas $d' = d / \bar d$:

$$
s = \frac{1}{1 + d'}, \qquad \text{RTM con pesos } \operatorname{softmax}_g\big(-\kappa\, d'(u,g)\big)
$$

- $\bar d$ es la media, **sin gradiente**, de todas las distancias galería–probe (registrados y no registrados) del fold y la parte.
- $\kappa$ es `rtm_softmax_scale` = 10.
- Con `rank_self_term: false`, el rango suave resta el término de la propia plantilla (0.5), así que un genuino sin impostores por delante tiene $\tilde r \approx 0$.

**Qué hace:** hace que la IDL y la RTM no dependan de la escala del embedding. Las pendientes se recalibran ($\alpha = 20$, $\beta = 1$, $\gamma = 30$) para que $S_{id}$ y $S_{det}$ no se saturen. Config `SwinGait3D_Su2024_Gait3D_OpenSetLoss_Calib.yaml`.

### 5.3 Escala de referencia genuina (`scale_reference: genuine`, `scale_detach: false`)

$$
d' = \frac{d}{\bar d_{gen}}, \qquad \bar d_{gen} = \frac{1}{|\mathcal{P}_M|}\sum_{x\in\mathcal{P}_M} d(x, g_{y(x)})
$$

Normaliza por la distancia genuina media en lugar de por la media de todos los pares, que está dominada por impostores. Así los genuinos quedan en $d' \approx 1$ y los intrusos por encima, y la RTM conserva margen que empujar.

Con `scale_detach: false` el gradiente atraviesa $\bar d_{gen}$. La pérdida es entonces exactamente invariante a la escala y su gradiente no tiene componente radial, de modo que no puede inflar las distancias.

Config `SwinGait3D_Su2024_Gait3D_OpenSetLoss_GenNorm_picasso.yaml` (E4.6), con los parámetros de 5.2 más estas dos opciones.

## 6. Open-pair loss (`OpenPairLoss`, propia)

Implementación en `losses/open_pair.py`. Con la simulación de la sección 2, en cada fold:

$$
\mathcal{L}_{\text{open-pair}} = \frac{1}{|\mathcal{A}|}\sum_{x_m \in \mathcal{P}_M}\sum_{x_u \in \mathcal{P}_{NM}}
\max\Big(0,\; \underbrace{d\big(x_m, g_{y(x_m)}\big)}_{d_{gen}} - \underbrace{\min_{g\in G} d(x_u, g)}_{d_{nm}} + m_{open}\Big)
$$

- $\mathcal{A}$ son los pares activos (hinge $> 0$), con `reduction: nonzero` (por defecto).
- Con `reduction: mean` se promedia sobre todos los pares $|\mathcal{P}_M|\,|\mathcal{P}_{NM}|$ (la fórmula literal).
- $m_{open} = 0.2$, el mismo margen y el mismo espacio euclídeo que la triplet.
- $\lambda = 0.5$.

**Qué hace:** exige que **toda** distancia genuina quede al menos $m_{open}$ por debajo de la distancia de **todo** intruso a su plantilla más cercana. Con eso existe un único umbral global $\tau$ que acepta los genuinos y rechaza los intrusos. Es un sustituto por pares del AUC entre las dos distribuciones que fijan FNIR y FPIR.

**Diferencias con la OpenSetLoss:**
- opera directamente sobre distancias, sin la transformación $1/(1+d)$ ni sigmoides con pendientes que calibrar;
- solo compara genuinos contra el intruso más cercano, que es lo que decide el rechazo con un umbral sobre el top-1.

Registra:
- `active_frac`: fracción de pares activos;
- `d_gen`: media de $d_{gen}$;
- `d_nm_min`: media de $d_{nm}$.

Configs:
- `GaitBase_Su2024_Gait3D_OpenPair.yaml`;
- `SwinGait3D_Su2024_Gait3D_OpenPair.yaml`;
- `GaitBase_GREW_OpenPair_picasso.yaml`.

### 6.1 Banco de memoria (`memory_bank: true`)

En el batch, $G$ tiene unas 24 plantillas, mientras que en test hay unas 800. El mínimo sobre pocas plantillas sobrestima la distancia al intruso más cercano. Con GaitBase a 60k en Gait3D:

| Galería | Distancia al intruso más cercano |
|---|---|
| 24 plantillas del batch | 1.37 |
| 2999 identidades de entrenamiento | 1.09 |
| Test | 1.10 |

Por eso el hinge se satisface en entrenamiento y no en test. El banco amplía la galería de los intrusos con una plantilla $b_j$ por identidad de entrenamiento:

$$
d_{nm}(x_u) = \min\Big(\min_{g\in G} d(x_u, g),\; \min_{j \ne y(x_u),\ j \text{ vista}} d(x_u, b_j)\Big)
$$

$$
b_j \leftarrow \mu\, b_j + (1-\mu)\, \frac{1}{|B_j|}\sum_{x\in B_j} f(x) \quad\text{(cada vez que } j \text{ aparece en el batch; la primera vez } b_j = \text{media)}
$$

- $B_j$ son las secuencias de $j$ en el batch.
- $\mu$ es `bank_momentum`, 0.5. Las apariciones pasadas pesan $0.5, 0.25, 0.125, \dots$
- Un $\mu$ alto promedia más secuencias pero arrastra embeddings más antiguos.

El banco no recibe gradiente y se actualiza después de calcular la pérdida. No se guarda en el checkpoint. El término genuino sigue usando la plantilla del batch.

Parámetros: `num_classes` (número de identidades de entrenamiento) y `bank_momentum`. Registra:
- `d_nm_min_batch`: mínimo solo con plantillas del batch;
- `nm_from_bank`: fracción de intrusos cuyo mínimo da el banco;
- `bank_filled`.

Config `GaitBase_Su2024_Gait3D_OpenPair_Bank.yaml`.

**Estado:** probado en Gait3D con GaitBase y $\mu = 0.5$, el banco no mejora el FNIR@1%FPIR y degrada el conjunto cerrado.
- Hasta unas 40k iteraciones el banco apenas interviene: con learning rate alto, las plantillas quedan obsoletas respecto al modelo actual.
- Cuando se activa, el espacio se comprime.

### 6.2 Escala genuina y cola de intrusos (OpenPair v2)

Dos opciones independientes, desactivadas por defecto.

**Escala genuina** (`scale_reference: genuine`, `scale_detach: false`). Con distancias en bruto y margen absoluto, la red puede cumplir el margen estirando o encogiendo el espacio sin separar mejor las clases. En cada fold y parte, todas las distancias se dividen por la media de las genuinas:

$$
s = \frac{1}{|\mathcal{P}_M|}\sum_{x_m} d(x_m, g_{y(x_m)}), \qquad
\max\big(0,\; d_{gen}/s - d_{nm}/s + m_{open}\big)
$$

$m_{open}$ pasa a ser un margen relativo. Con gradiente a través de $s$, la pérdida es exactamente invariante a la escala del embedding y no tiene componente radial, como la OSL de E4.6. Registra `scale` ($s$); `d_gen` y `d_nm_min` siguen en unidades brutas.

**Cola de intrusos** (`tail_frac: β`). El FNIR@1%FPIR lo fija el ~1 % de intrusos más cercanos, mientras que la media sobre todos los pares es un sustituto del AUC completo. Con GaitBase, de 40k a 60k mejoran el AUC, el OSCR y el FNIR@5 %, pero empeora el FNIR@1 %. En cada fold y parte solo entran en la hinge los $\lceil \beta\,|\mathcal{P}_{NM}| \rceil$ intrusos con menor $d_{nm}$:

$$
\frac{1}{|\mathcal{A}|}\sum_{x_m \in \mathcal{P}_M}\ \sum_{x_u \in \mathcal{T}_\beta}
\max\big(0,\; d_{gen}(x_m) - d_{nm}(x_u) + m_{open}\big)
$$

donde $\mathcal{T}_\beta$ son esos intrusos más cercanos. Es el sustituto estándar del AUC parcial con FPR ∈ [0, β] (Narasimhan & Agarwal, ICML 2013). Con β = 0.25 y 8 sujetos por fold quedan 8 de 32 intrusos.

Config `GaitBase_Su2024_Gait3D_OpenPair_v2.yaml`: las dos opciones, m = 0.2 y β = 0.25. **Estado:** en curso.

### 6.3 Intruso más cercano entre secuencias (`nm_sequences: true`, v2.a)

Corrige el sesgo de valor extremo de la sección 6.1 sin banco. El intruso busca su elemento de galería más cercano entre las plantillas del fold **y** todas las secuencias individuales de los sujetos conocidos del batch $S_K$ (las de plantilla y las de probe registrado):

$$
d_{nm}(x_u) = \min\Big(\min_{g\in G} d(x_u, g),\; \min_{x\in S_K} d(x_u, x)\Big)
$$

- Hay unas 4 veces más candidatos. Todos se calculan con el modelo actual y con gradiente, así que nada queda obsoleto (al contrario que con el banco).
- $S_K$ excluye a los sujetos del fold, así que la propia identidad del intruso nunca es candidata.
- El término genuino sigue usando la plantilla del batch.
- Se aplica antes de normalizar por la escala genuina y de seleccionar la cola.
- Registra `d_nm_min_tmpl` (solo plantillas) y `nm_from_seq` (fracción de intrusos cuyo mínimo es una secuencia).
- Limitación: en test la galería son plantillas medias. Una secuencia suelta es más ruidosa que una media, así que su mínimo no es del todo comparable.

Config `GaitBase_Su2024_Gait3D_OpenPair_v2a.yaml` (v2 + `nm_sequences: true`). **Estado:** en curso.

## 7. Resumen

| Pérdida | Fichero | Opera sobre | Objetivo |
|---|---|---|---|
| Triplet | `triplet.py`, `triplet_su.py` | distancias, tripletas del batch | separación relativa por ancla (conjunto cerrado) |
| CE + BNNeck | `ce.py` | logits coseno | clasificación de las identidades de entrenamiento |
| OpenSetLoss (IDL + RTM) | `open_set.py` | puntuaciones $1/(1+d)$, simulación open-set | DIR@FPIR diferenciable + bajar el umbral de intrusos |
| — sin identificación | `open_set.py` | ídem | solo detección (genuino sobre umbrales de intrusos) |
| — calibrada / escala genuina | `open_set.py` | $d' = d/\bar d$ o $d/\bar d_{gen}$ | lo mismo, invariante a la escala del embedding |
| OpenPair | `open_pair.py` | distancias, simulación open-set | genuino $+\, m$ < intruso más cercano (umbral global) |
| — banco de memoria | `open_pair.py` | ídem + plantillas EMA de todas las identidades | intruso más cercano realista (galería grande) |
| — v2: escala genuina + cola | `open_pair.py` | $d/\bar d_{gen}$, intruso entre el β más cercano | margen relativo; AUC parcial en FPIR baja |
| — v2.a: + secuencias como candidatos | `open_pair.py` | ídem, $d_{nm}$ sobre plantillas ∪ secuencias conocidas | intruso más cercano más realista sin banco |
