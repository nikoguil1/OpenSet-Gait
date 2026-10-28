# Directrices de Proyecto: OpenSet-Gait

## 1. Contexto de ClickUp
- **Workspace ID**: `9015482533`
- **Espacio**: Investigación (Space ID `90151609459`)
- **Proyecto**: OpenSet_Gait (Folder ID `90152375348`)
- **Épica raíz**: `CU-123nktjx7p5` (evaluación open-set de gait en Gait3D, SUSTech1K, CCGR y CCPG)
- **Lista de Tareas** (`📝 Tareas`):
  - List ID: `901503756065`
  - Enlace: https://app.clickup.com/9015482533/v/li/901503756065
- **Doc de Métricas (Resultados)**:
  - Doc ID: `8cnup55-6135`
  - Page ID: `8cnup55-1935`
  - Enlace: https://app.clickup.com/9015482533/v/dc/8cnup55-6135/8cnup55-1935
- **Doc de Bitácora (Anotaciones)**:
  - Doc ID: `8cnup55-6155`
  - Page ID: `8cnup55-1955`
  - Enlace: https://app.clickup.com/9015482533/v/dc/8cnup55-6155/8cnup55-1955
- **Estados de Tareas**:
  - `IDEA`: Hipótesis teóricas, borradores, lecturas pendientes y brainstorming conceptual.
  - `PLANIFICADA`: Tareas con alcance técnico cerrado, etiquetas y criterios de aceptación definidos, listas para ser tomadas.
  - `EN DESARROLLO`: Tarea activa de implementación en código.
  - `ENTRENANDO / EVALUANDO`: Tarea con scripts o entrenamientos corriendo en servidor/GPU.
  - `DESCARTADO`: Hipótesis invalidadas empíricamente (requiere registro de causa en Doc Anotaciones).
  - `COMPLETADO`: Código validado, pruebas superadas, commit/push realizado y tabla Resultados actualizada.

---

## 2. Taxonomía de Etiquetas y Prioridades
- **Etiquetas obligatorias al crear tareas**:
  - `#data`: Pretratamiento de siluetas/pose, data loaders, particiones train/test y particiones open-set conocidos/desconocidos.
  - `#model`: Arquitectura neuronal (backbones de gait, cabezales, funciones de pérdida métricas) y puntuaciones de rechazo de intrusos.
  - `#infra`: Scripts de entrenamiento, gestión de memoria GPU, profiling, exportación.
  - `#paper`: Curvas ROC, matrices de confusión, tablas comparativas, figuras y redacción.
- **Criterio de Prioridad**:
  - `Urgente` (Roja): Errores que rompen el pipeline o causan data leakage / corrupción de métricas.
  - `Alta` (Amarilla): Tareas que desbloquean un entrenamiento planificado en el servidor.
  - `Normal` (Azul): Refactorizaciones, modularización y pruebas unitarias.
  - `Baja` (Gris): Ideas secundarias, lecturas de papers o refactors menores.

---

## 3. Jerarquía y Barridos de Experimentos
- **Experimentos múltiples**: Cuando se proponga evaluar varios hiperparámetros (ej. distintas pérdidas, márgenes $m$, temperaturas $T$ o datasets), no crear tareas principales sueltas.
  - Crear una tarea padre: `[EXP] Nombre del experimento general`.
  - Crear una subtarea por cada variante: `Evaluar BHCn m=8`, `Evaluar BHCn m=10`, etc.

---

## 4. Entorno de Ejecución
- **Activación de entorno**: Antes de ejecutar cualquier script o test en la terminal remota, activar el entorno:
  `source /home/nico/gait/OpenSet-Gait/.venv/bin/activate` (el venv se crea en CU-123nktjx7yj; el de `~/gait/OpenGait/.venv` está roto).
- **Datos**: `/home/GAIT_local/SSD_grande` (Gait3D, SUSTech1K, CCPG, GREW, CASIA-B…). Nunca escribir en las carpetas de datos originales.
- Todas las comprobaciones sintácticas, pruebas unitarias y ejecuciones preliminares deben correr estrictamente en este entorno.

---

## 5. Política Estricta de Git, Commits y Push
- **PROHIBIDO hacer commit o push automáticamente**: Desarrolla, refactoriza y prueba libremente, pero no ejecutes `git commit` ni `git push` sin orden explícita del usuario (ej. *"haz commit y push"*, *"cierra tarea y sube cambios"*).
- **Control de artefactos**: Antes de preparar el commit, comprueba con `git status` que no haya ficheros binarios (`.pt`, `.ckpt`, `.onnx`, dumps de datos ni logs masivos). Evitar `git add .` indiscriminado.
- **Gestión de ramas**:
  - Trabajo principal directamente sobre `main`.
  - Ramas experimentales solo si un cambio rompe el pipeline activo (`exp/CU-[id]-nombre`).
  - Merge obligatorio a `main` y borrado de la rama en cuanto el código esté verificado y entrene sin fallos.
- **Formato obligatorio de Commit**:
  ```text
  <type>(<scope>): [CU-id] Título conciso y descriptivo

  MOTIVACIÓN:
  - Justificación teórica o técnica del cambio en datos, modelo o protocolo de evaluación.

  DETALLE DE CAMBIOS:
  - Archivos y lógica alterada (pretratamiento, backbone, pérdidas, evaluador open-set, etc.).
  - Hiperparámetros modificados.

  VERIFICACIÓN:
  - Pruebas sintácticas, unitarias o épocas iniciales ejecutadas en GPU.
  ```

## 6. Flujo de Estados y Documentación

### 6.1. Ciclo de Vida de las Tareas
1. **Fase Conceptual (`IDEA` ➔ `PLANIFICADA`)**:
   - **Disparador**: Solicitud explícita del usuario de estructurar, aterrizar o desglosar una idea.
   - **Acción**:
     - Leer la descripción y los comentarios de la tarea en `IDEA`.
     - Extraer requisitos técnicos y crear una o varias tareas en `PLANIFICADA` con sus etiquetas (`#data`, `#model`, etc.), objetivos y checklist de aceptación.
     - Vincular el ID de la idea origen en la descripción (ej. *"Derivada de CU-xxx"*).
     - Comentar en la tarea `IDEA` el desglose generado y moverla a `COMPLETADO` (o mantenerla como contenedor si es una épica).
2. **Inicio del trabajo (`PLANIFICADA` ➔ `EN DESARROLLO`)**:
   - Al tomar una tarea planificada para programar, mover su estado a `EN DESARROLLO`.
3. **Ejecución en servidor (`EN DESARROLLO` ➔ `ENTRENANDO / EVALUANDO`)**:
   - Si un proceso, entrenamiento o barrido queda corriendo en segundo plano o en cola de GPU, mover a este estado hasta su finalización.
4. **Gestión de fallos de ejecución (OOM / Divergencias / Bugs)**:
   - Si el script aborta por CUDA OOM, desajuste dimensional o NaNs durante la depuración, NO hacer commit.
   - Mantener el estado en `EN DESARROLLO` y registrar un comentario en la tarea con el stack trace y contexto técnico del error.
5. **Criterio de salida por descarte (`DESCARTADO`)**:
   - Mover aquí únicamente si una hipótesis teórica o arquitectura se descarta definitivamente tras fallar (inviabilidad matemática, degradación severa de métricas) o por indicación expresa del usuario.
   - PROHIBIDO commitear código roto o incompleto a `main`.
   - Acción obligatoria: detallar en un comentario de la tarea la causa técnica y registrar una entrada en el Doc `Anotaciones` (sección *Post-Mortem y Análisis de Fallos*).
6. **Cierre y entrega (`COMPLETADO`)**:
   - Comprobar que los puntos de la checklist técnica de la tarea estén validados:
     - [ ] Sintaxis y dimensiones de tensores comprobadas en entorno de ejecución.
     - [ ] Ausencia de fuga de datos (*data leakage*) en las particiones.
     - [ ] Épocas preliminares ejecutadas con convergencia estable.
     - [ ] Fila registrada en el Doc `Resultados`.
     - [ ] Commit atómico y push completado en `main`.
   - Solicitar confirmación explícita al usuario para commit y push.
   - Tras el push, mover la tarea a `COMPLETADO` y añadir un comentario final con el commit hash (`git rev-parse --short HEAD`) y el resumen de cambios.

---

### 6.2. Protocolo de Sincronización Documental
- **Doc `Resultados` (Métricas Dinámicas)**:
  - **Columnas de métricas**: Rank-1, mAP (cerrado); EER, TAR@1%FAR, TAR@0.1%FAR (verificación); DIR@1%FPIR, OSCR, AUROC (open-set). Con el protocolo unificado, media ± std entre particiones.
  - **Campos obligatorios por fila**: `Fecha` (`YYYY-MM-DD`), `Tarea ID` (`CU-xxx`), `Commit` (hash corto), `Dataset`, `Modelo`, `Pérdida`, `Protocolo` (`legacy 10 %` o `unificado`), valores numéricos obtenidos, `Ruta Checkpoint / Artefacto` (ruta relativa local en servidor, ej. `output/Gait3D/DeepGaitV2/<save_name>/checkpoints/...pt`) y `Notas / Conclusiones` (resumen en una sola frase).
- **Doc `Anotaciones` (Bitácora de Laboratorio)**:
  - Registrar hipótesis teóricas, justificaciones matemáticas, preprocesamiento o citas bibliográficas en `## 1. Hipótesis Teóricas y Decisiones de Diseño`.
  - Registrar análisis post-mortem y motivos de descarte en `## 2. Post-Mortem y Análisis de Fallos`.

## 7. Mantenimiento Autónomo de Arquitectura (`ARCHITECTURE.md`)
- **Archivo objetivo**: `ARCHITECTURE.md` en la raíz del repositorio.
- **Regla de ciclo de vida (Creación / Actualización)**:
  - **Si NO existe**: Créalo por primera vez con la estructura base completa (Diagrama `mermaid` con preprocesamiento, backbone y pérdidas, Tabla de tensores/shapes y Detalle de funciones de pérdida).
  - **Si YA existe**: Lee el archivo y actualiza únicamente las partes afectadas (nodos del diagrama Mermaid, filas de la tabla de dimensiones o descripción de losses) sin sobrescribir notas manuales previas.
- **Condición de disparo**: Cualquier cambio en el código que modifique:
  - Pretratamiento de entradas (siluetas, pose, resolución, número de frames) o protocolo de evaluación open-set.
  - Dimensiones intermedias o shapes de tensores.
  - Capas de la red, mecanismos de atención o cabezales de inferencia.
  - Funciones de pérdida (p. ej., Triplet, BHCn, SupCon, Cross-Entropy) o pesos de optimización ($\lambda$).
- **Integración con Git y ClickUp**:
  - `ARCHITECTURE.md` debe incluirse en el mismo commit que el código fuente cuando el usuario ordene `commit` + `push`.
  - En el comentario final de la tarea en ClickUp, añade una línea indicando: `[Doc] ARCHITECTURE.md generado/actualizado con la nueva topología.`
