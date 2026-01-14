#!/bin/bash
# Download checkpoints and logs from Oxford ARC
# Usage: ./scripts/arc/sync_from_arc.sh [username]

USER=${1:-youdar}
ARC_HOST="arc-jump"  # Uses SSH config alias (works without VPN)
REMOTE_DIR="/data/coml-prog-synthesis/${USER}"
LOCAL_DIR="/Users/youdar/GitHub/synthstats"

echo "Downloading from ${ARC_HOST}..."

# Download checkpoints
echo "Syncing checkpoints..."
mkdir -p ${LOCAL_DIR}/checkpoints
rsync -avz --progress \
    ${USER}@${ARC_HOST}:${REMOTE_DIR}/checkpoints/ \
    ${LOCAL_DIR}/checkpoints/

# Download logs
echo "Syncing logs..."
mkdir -p ${LOCAL_DIR}/logs
rsync -avz --progress \
    ${USER}@${ARC_HOST}:${REMOTE_DIR}/logs/ \
    ${LOCAL_DIR}/logs/

# Download wandb runs (offline mode)
echo "Syncing wandb runs..."
mkdir -p ${LOCAL_DIR}/wandb
rsync -avz --progress \
    ${USER}@${ARC_HOST}:${REMOTE_DIR}/wandb/ \
    ${LOCAL_DIR}/wandb/

echo "Download complete!"
echo ""
echo "To sync wandb runs online:"
echo "  cd ${LOCAL_DIR}/wandb && wandb sync ."
