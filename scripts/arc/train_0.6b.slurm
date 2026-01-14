#!/bin/bash
#SBATCH --job-name=synth_0.6b
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/data/coml-prog-synthesis/%u/logs/synth_0.6b_%j.out
#SBATCH --error=/data/coml-prog-synthesis/%u/logs/synth_0.6b_%j.err
#SBATCH --mail-type=NONE
# To enable email notifications, submit with: sbatch --mail-type=END,FAIL --mail-user=YOUR_EMAIL ...

# SynthStats training with Qwen3-0.6B on Oxford ARC
# Submit with: sbatch scripts/arc/train_0.6b.slurm

set -e

echo "=== Job Info ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start time: $(date)"
echo ""

# Load modules (Python module optional; venv provides Python)
PYTHON_MODULE=${PYTHON_MODULE:-""}
module purge
if [ -n "${PYTHON_MODULE}" ]; then
    module load ${PYTHON_MODULE}
fi
module load CUDA/12.0.0

# Activate virtual environment
source /data/coml-prog-synthesis/${USER}/venvs/synthstats/bin/activate

# Set environment variables
export HF_HOME=/data/coml-prog-synthesis/${USER}/huggingface
export TORCH_HOME=/data/coml-prog-synthesis/${USER}/torch
export TRANSFORMERS_CACHE=${HF_HOME}
export WANDB_MODE=offline
export WANDB_DIR=/data/coml-prog-synthesis/${USER}/wandb

# Project directory
PROJECT_DIR=/data/coml-prog-synthesis/${USER}/synthstats
cd ${PROJECT_DIR}

# Checkpoint directory
CKPT_DIR=/data/coml-prog-synthesis/${USER}/checkpoints/${SLURM_JOB_NAME}
mkdir -p ${CKPT_DIR}

# Find latest checkpoint if exists for resumption
LATEST_CKPT=$(ls -t ${CKPT_DIR}/checkpoint_*.pt 2>/dev/null | head -1 || true)
RESUME_ARG=""
if [ -n "${LATEST_CKPT}" ]; then
    echo "Resuming from: ${LATEST_CKPT}"
    RESUME_ARG="resume_from=${LATEST_CKPT}"
fi

echo "=== Starting Training ==="
echo "Model: Qwen3-0.6B"
echo "Checkpoint dir: ${CKPT_DIR}"
echo ""

# Force unbuffered output for real-time logging
export PYTHONUNBUFFERED=1

# Run training (use python directly since venv is activated)
python -u scripts/train_skyrl.py \
    model=qwen3_0.6b \
    trainer=subtb \
    trainer.num_episodes=1000 \
    trainer.batch_size=4 \
    ++output_dir=${CKPT_DIR} \
    +checkpoint_interval=50 \
    +cluster=oxford_arc \
    ${RESUME_ARG}

echo ""
echo "=== Training Complete ==="
echo "End time: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
