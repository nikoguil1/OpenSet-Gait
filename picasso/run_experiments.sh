#!/bin/bash -l
# Slurm launcher for OpenSet-Gait on picasso (SCBI, UMA), adapted from the run_experiments.sh template.
#
#   cd ~/fscratch/OpenSetGait/jobs && sbatch ~/OpenSetGait/picasso/run_experiments.sh             # train (defaults)
#
# NEVER use sbatch --export on picasso: jobs submitted with it are cancelled by the system 1-2 s after starting
# (no log), and since 2026-11-20 the sbatch wrapper rejects it. For another config / dataset / phase, write a
# small job script in ~/fscratch/OpenSetGait/jobs with its own #SBATCH lines (they override the ones below) and
#   export DATASET=GREW                                    # Gait3D | GREW
#   export CFG_FILE=configs/...yaml                        # relative to the repository root
#   export PHASE=test                                      # optional (default train)
#   bash $HOME/OpenSetGait/picasso/run_experiments.sh
# then `sbatch job_X.sh` (examples: job_GaitBase_GREW_OpenPair.sh, job_Scaling_SwinGait3D_*xB200.sh).
#
# - Training resumes from the last checkpoint in output/<dataset>/<model>/<save_name>/checkpoints (if any),
#   so a job killed by the time limit is continued by submitting the same command again.
# - The dataset is a .tar.gz in fscratch (one file, to stay within the fscratch file quota), extracted once
#   per node into $LOCALSCRATCH (flock + marker file).
# - The config may use the placeholder DATASET_ROOT in data_cfg.dataset_root; it is replaced by the local copy.
# - Python packages missing from the pytorch module (kornia, timm, opencv, imageio) live in PYTHONUSERBASE
#   (installed once with: module load pytorch/2.10.0; PYTHONUSERBASE=... python -m pip install --user ...).
#SBATCH -J E4.6_GenNorm
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --mem=400G
#SBATCH --ntasks-per-node=1          # one torchrun process, which spawns one worker per GPU
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --constraint=dgx             # A100 (exa01-04) or B200 (blk01-02) nodes, 8 GPUs each
#SBATCH --output=log-%x-%J.out
#SBATCH --error=log-%x-%J.err

set -eo pipefail   # no -u: module scripts use unset variables
echo "NODELIST=${SLURM_NODELIST}  JOB=${SLURM_JOB_ID}  $(date '+%F %T')"

# ---- Configuration ----
CODE_DIR="$HOME/OpenSetGait"
CFG_FILE="${CFG_FILE:-configs/swingait/SwinGait3D_Su2024_Gait3D_OpenSetLoss_GenNorm_picasso.yaml}"  # relative to CODE_DIR
PHASE="${PHASE:-train}"
DATASET="${DATASET:-Gait3D}"         # Gait3D | GREW (selects the archive below)
case "$DATASET" in
    Gait3D) DATASET_TAR="$HOME/fscratch/dataset/Gait3D/Gait3D-sils-64-44-pkl.tar.gz"
            DATASET_NFILES=25309 ;;  # files in the extracted dataset (extraction check)
    GREW)   DATASET_TAR="$HOME/fscratch/dataset/GREW/GREW-pkl.tar.gz"
            DATASET_NFILES=126887 ;;
    *)      echo "Unknown DATASET=$DATASET"; exit 1 ;;
esac
export PYTHONUSERBASE="$HOME/fscratch/pyenv/pytorch-2.10.0"
# -----------------------

N_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | grep -c .)
echo "Detected $N_GPUS GPUs (${CUDA_VISIBLE_DEVICES})"
nvidia-smi -L

# ---- Dataset in the node's local scratch ----
DATASET_BASE_DIR=$(basename "$DATASET_TAR" | sed -E 's/\.tar(\.gz)?$//')   # top folder inside the archive
MYLOCALSCRATCH="$LOCALSCRATCH/$USER"
JOBLOCALSCRATCH="$MYLOCALSCRATCH/$SLURM_JOB_ID"
mkdir -p "$JOBLOCALSCRATCH"
LOCK_FILE="$MYLOCALSCRATCH/dataset_extraction.lock"
MARKER_FILE="$MYLOCALSCRATCH/.extraction_done_${DATASET_BASE_DIR}_$(stat -c %Y "$DATASET_TAR")"  # new tar -> re-extract
DATASET_ROOT="$MYLOCALSCRATCH/$DATASET_BASE_DIR"

(
    echo "Waiting for extraction lock (if another job is extracting)..."
    flock -x 9
    if [ ! -f "$MARKER_FILE" ] || [ "$(find "$DATASET_ROOT" -type f 2>/dev/null | wc -l)" -ne "$DATASET_NFILES" ]; then
        echo "Extracting $DATASET_TAR into $MYLOCALSCRATCH ..."
        rm -rf "$DATASET_ROOT" "$MYLOCALSCRATCH"/.extraction_done_"${DATASET_BASE_DIR}"_*
        case "$DATASET_TAR" in
            *.gz) tar -I pigz -xf "$DATASET_TAR" -C "$MYLOCALSCRATCH/" ;;   # gzip archive (pigz: faster gunzip)
            *)    tar -xf "$DATASET_TAR" -C "$MYLOCALSCRATCH/" ;;
        esac
        [ "$(find "$DATASET_ROOT" -type f | wc -l)" -eq "$DATASET_NFILES" ] || { echo "Extraction incomplete"; exit 1; }
        touch "$MARKER_FILE"
        echo "Extraction complete."
    else
        echo "Dataset already extracted and validated by a previous job."
    fi
) 9> "$LOCK_FILE"

# ---- Environment ----
module load pytorch/2.10.0
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1

cd "$CODE_DIR"   # dataset_partition and output/ are relative to the repository root

# ---- Job config: dataset root and (train) resume point ----
CFG_NAME=$(basename "$CFG_FILE")
JOB_CFG="$JOBLOCALSCRATCH/$CFG_NAME"
sed "s|DATASET_ROOT|$DATASET_ROOT|" "$CFG_FILE" > "$JOB_CFG"

if [ "$PHASE" = "train" ]; then
    SAVE_NAME=$(python -c "import yaml,sys; c=yaml.safe_load(open(sys.argv[1])); print(c['trainer_cfg']['save_name'])" "$JOB_CFG" | tail -1)
    # numeric sort on the iteration (a plain sort puts -90000 after -100000)
    LAST_CKPT=$(ls output/*/*/"$SAVE_NAME"/checkpoints/"$SAVE_NAME"-*.pt 2>/dev/null \
                | awk -F- '{n=$NF; sub(/\.pt$/,"",n); print n+0, $0}' | sort -n | tail -1 | cut -d' ' -f2- || true)
    if [ -n "$LAST_CKPT" ]; then
        RESTORE=$((10#$(basename "$LAST_CKPT" .pt | sed 's/.*-//')))
        # Resume from our own checkpoint: trainer_cfg.restore_hint = iteration (whatever it was: 0 or a path to an
        # init checkpoint, e.g. DeepGaitV2 for SwinGait stage 2) and keep the saved optimizer/scheduler state.
        python - "$JOB_CFG" "$RESTORE" <<'PYEOF'
import sys, yaml
path, it = sys.argv[1], int(sys.argv[2])
cfg = yaml.safe_load(open(path))
cfg['trainer_cfg']['restore_hint'] = it
cfg['trainer_cfg']['optimizer_reset'] = False
cfg['trainer_cfg']['scheduler_reset'] = False
cfg['trainer_cfg']['restore_ckpt_strict'] = True
yaml.safe_dump(cfg, open(path, 'w'), sort_keys=False)
PYEOF
        echo "Resuming from $LAST_CKPT (iteration $RESTORE)"
    fi
fi
grep -n "dataset_root\|restore_hint\|save_name" "$JOB_CFG"

# ---- Run ----
echo "Running $PHASE with $CFG_FILE on $N_GPUS GPUs"
time python -m torch.distributed.run \
    --standalone \
    --nnodes 1 \
    --nproc_per_node "$N_GPUS" \
    opengait/main.py \
    --cfgs "$JOB_CFG" \
    --phase "$PHASE" --log_to_file

rm -rf "$JOBLOCALSCRATCH"   # the extracted dataset stays in $MYLOCALSCRATCH for later jobs on this node
echo "Done $(date '+%F %T')"
