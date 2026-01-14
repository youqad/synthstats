#!/bin/bash
#SBATCH --job-name=synth_tests
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/data/coml-prog-synthesis/%u/logs/tests_%j.out
#SBATCH --error=/data/coml-prog-synthesis/%u/logs/tests_%j.err

# Run SynthStats tests on Oxford ARC
# Submit with: sbatch scripts/arc/test.slurm

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

# Activate virtual environment
source /data/coml-prog-synthesis/${USER}/venvs/synthstats/bin/activate

# Project directory
PROJECT_DIR=/data/coml-prog-synthesis/${USER}/synthstats
cd ${PROJECT_DIR}

export PYTHONUNBUFFERED=1

echo "=== Installing dev extras ==="
uv pip install -e ".[dev]"

echo "=== Running tests ==="
pytest

echo ""
echo "=== Tests Complete ==="
echo "End time: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
