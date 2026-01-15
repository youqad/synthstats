#!/bin/bash
#SBATCH --job-name=synth_smoke
#SBATCH --partition=devel
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/data/coml-prog-synthesis/%u/logs/synth_smoke_%j.out
#SBATCH --error=/data/coml-prog-synthesis/%u/logs/synth_smoke_%j.err
#SBATCH --mail-type=NONE

# Small smoke test for SynthStats SkyRL training on Oxford ARC
# Submit with: sbatch scripts/arc/train_smoke.slurm

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
# Keep CUDA module unloaded for CPU-only smoke run

# Activate virtual environment
source /data/coml-prog-synthesis/${USER}/venvs/synthstats/bin/activate

# Set environment variables
export HF_HOME=/data/coml-prog-synthesis/${USER}/huggingface
export TORCH_HOME=/data/coml-prog-synthesis/${USER}/torch
export TRANSFORMERS_CACHE=${HF_HOME}
export WANDB_MODE=offline
export WANDB_DIR=/data/coml-prog-synthesis/${USER}/wandb
export CUDA_VISIBLE_DEVICES=""

# Project directory
PROJECT_DIR=/data/coml-prog-synthesis/${USER}/synthstats
cd ${PROJECT_DIR}

# Ensure local package is importable even if not installed
export PYTHONPATH=${PROJECT_DIR}:${PYTHONPATH}

# Ensure CPU-only torch to avoid CUDA ABI mismatch on compute nodes
echo "=== Installing CPU torch ==="
python -m pip install --upgrade --force-reinstall "torch==2.5.1+cpu" --index-url https://download.pytorch.org/whl/cpu

# Ensure base dependencies are installed in the venv
echo "=== Installing synthstats dependencies (editable) ==="
python -m pip install -e .

# Checkpoint directory
CKPT_DIR=/data/coml-prog-synthesis/${USER}/checkpoints/${SLURM_JOB_NAME}
mkdir -p ${CKPT_DIR}

echo "=== Starting Smoke Test ==="
echo "Model: Qwen3-0.6B"
echo "Checkpoint dir: ${CKPT_DIR}"
echo ""

# Force unbuffered output for real-time logging
export PYTHONUNBUFFERED=1

# Minimal run to sanity-check collection and reward flow
python -u scripts/train_skyrl.py \
    model=mock \
    trainer=subtb \
    trainer.num_episodes=1 \
    trainer.batch_size=1 \
    trainer.replay_buffer_size=0 \
    trainer.replay_ratio=0.0 \
    ++output_dir=${CKPT_DIR} \
    +checkpoint_interval=1 \
    +cluster=oxford_arc

echo ""
echo "=== Smoke Test Complete ==="
echo "End time: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
