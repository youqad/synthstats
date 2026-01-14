#!/bin/bash
# Sync project to Oxford ARC
# Usage: ./scripts/arc/sync_to_arc.sh [username]

REMOTE_USER=${1:-}
ARC_HOST="arc-jump"  # Uses SSH config alias (works without VPN)
if [ -z "${REMOTE_USER}" ]; then
    REMOTE_USER=$(ssh -o BatchMode=yes ${ARC_HOST} "whoami")
    RSYNC_TARGET="${ARC_HOST}"
else
    RSYNC_TARGET="${REMOTE_USER}@${ARC_HOST}"
fi
REMOTE_DIR="/data/coml-prog-synthesis/${REMOTE_USER}/synthstats"

echo "Syncing to ${RSYNC_TARGET}:${REMOTE_DIR}"

rsync -avz --progress \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.venv' \
    --exclude='.venv*' \
    --exclude='.hypothesis' \
    --exclude='*.egg-info' \
    --exclude='outputs/' \
    --exclude='checkpoints/' \
    --exclude='wandb/' \
    --exclude='.mypy_cache' \
    --exclude='.ruff_cache' \
    --exclude='.pytest_cache' \
    --exclude='htmlcov/' \
    --exclude='.coverage' \
    --exclude='uv.lock' \
    /Users/youdar/GitHub/synthstats/ \
    ${RSYNC_TARGET}:${REMOTE_DIR}/

echo "Sync complete!"
echo ""
echo "Next steps:"
echo "  1. SSH to ARC: ssh ${ARC_HOST}"
echo "  2. Setup env (first time): bash ${REMOTE_DIR}/scripts/arc/setup_env.sh"
echo "  3. Submit job: cd ${REMOTE_DIR} && sbatch scripts/arc/train_0.6b.slurm"
