#!/bin/bash
#SBATCH --job-name=synth_setup
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/data/coml-prog-synthesis/%u/logs/setup_%j.out
#SBATCH --error=/data/coml-prog-synthesis/%u/logs/setup_%j.err

# Environment setup for SynthStats on Oxford ARC
# Submit with: sbatch scripts/arc/setup_env.slurm

set -e

echo "=== Setting up SynthStats environment on Oxford ARC ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo ""

# Create directory structure
USER_DIR=/data/coml-prog-synthesis/${USER}
mkdir -p ${USER_DIR}/{synthstats,checkpoints,logs,huggingface,torch,wandb,venvs,cache}
mkdir -p ${USER_DIR}/cache/{pip,uv}

echo "Created directories in ${USER_DIR}"

# Keep caches off home (quota)
export XDG_CACHE_HOME=${USER_DIR}/cache
export UV_CACHE_DIR=${USER_DIR}/cache/uv
export PIP_CACHE_DIR=${USER_DIR}/cache/pip
export RAYON_NUM_THREADS=1
export UV_NUM_THREADS=1

# Load modules
# Python module is only used to bootstrap uv; the venv uses Python 3.12.
PYTHON_MODULE=${PYTHON_MODULE:-Python/3.11.3-GCCcore-12.3.0}
PYTHON_VERSION=${PYTHON_VERSION:-3.12}
CUDA_MODULE=${CUDA_MODULE:-CUDA/12.0.0}
module purge
if [ -n "${PYTHON_MODULE}" ]; then
    module load ${PYTHON_MODULE}
fi
module load ${CUDA_MODULE}

echo "Loaded modules: ${PYTHON_MODULE}, ${CUDA_MODULE}"

# Install uv (used to build a Python 3.12 venv)
echo "Installing uv..."
python -m pip install --upgrade pip
python -m pip install --user uv
export PATH="${HOME}/.local/bin:${PATH}"
UV_BIN=${UV_BIN:-${HOME}/.local/bin/uv}

# Create virtual environment
VENV_DIR=${USER_DIR}/venvs/synthstats
if [ -d "${VENV_DIR}" ]; then
    echo "Removing existing venv..."
    rm -rf ${VENV_DIR}
fi

echo "Creating virtual environment at ${VENV_DIR} with Python ${PYTHON_VERSION}..."
${UV_BIN} venv --python ${PYTHON_VERSION} --seed ${VENV_DIR}
source ${VENV_DIR}/bin/activate

# Install PyTorch with CUDA wheels (override with TORCH_CUDA_INDEX, e.g. cu128)
echo "Installing PyTorch..."
TORCH_CUDA_INDEX=${TORCH_CUDA_INDEX:-cu121}
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/${TORCH_CUDA_INDEX}

# Install synthstats with extras using uv pip install
cd ${USER_DIR}/synthstats
echo "Installing synthstats with extras (ml, training, skyrl, pymc)..."
uv pip install -e ".[ml,training,skyrl,pymc]"

echo ""
echo "=== Verification ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import synthstats; print('synthstats: OK')"
python -c "import hydra_plugins.hydra_submitit_launcher; print('submitit launcher: OK')"

echo ""
echo "=== Setup complete! ==="
echo "Activate with: source ${VENV_DIR}/bin/activate"
echo "Submit training with: sbatch scripts/arc/train_0.6b.slurm"
