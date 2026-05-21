#!/usr/bin/env bash
# Launch nano-diffusion on a Prime Intellect GPU instance.
#
# Setup:
#   1. Reserve an instance at app.primeintellect.ai (A100 / H100)
#   2. Copy the SSH host from the dashboard (Instances → Connect)
#   3. Export PI_HOST before running, or edit the default below:
#        export PI_HOST="ubuntu@123.45.67.89"
#   4. Ensure your SSH key is registered with the instance
#   5. Export WANDB_API_KEY in your local shell
#
# Usage:
#   ./launchers/prime_intellect.sh
#   DATASET=mnist ./launchers/prime_intellect.sh
#   OVERRIDES="hidden_dim=512 depth=12" ./launchers/prime_intellect.sh
#   PI_HOST=ubuntu@1.2.3.4 DATASET=cifar10 ./launchers/prime_intellect.sh

set -euo pipefail

# ── config ───────────────────────────────────────────────────────────────────
PI_HOST="${PI_HOST:-ubuntu@<your-instance-ip>}"
REMOTE_DIR="~/nano-diffusion"
DATASET="${DATASET:-cifar10}"
OVERRIDES="${OVERRIDES:-}"
WANDB_API_KEY="${WANDB_API_KEY:-}"
SESSION="nano-diffusion"
# ─────────────────────────────────────────────────────────────────────────────

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [[ "$PI_HOST" == *"<your-instance-ip>"* ]]; then
    echo "error: set PI_HOST to your instance address (e.g. export PI_HOST=ubuntu@1.2.3.4)"
    exit 1
fi

if [[ -z "$WANDB_API_KEY" ]]; then
    echo "warning: WANDB_API_KEY is not set — wandb will run in offline mode"
fi

# ── 1. sync source ────────────────────────────────────────────────────────────
echo "→ syncing code to $PI_HOST:$REMOTE_DIR"
rsync -az --progress \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='checkpoints' \
    --exclude='launchers' \
    --exclude='*.pt' \
    "$ROOT/" "$PI_HOST:$REMOTE_DIR/"

# ── 2. install deps (idempotent) ──────────────────────────────────────────────
echo "→ installing dependencies"
ssh "$PI_HOST" "
    cd $REMOTE_DIR
    pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install -q datasets huggingface_hub einops omegaconf wandb Pillow tqdm
"

# ── 3. write env file (avoids key appearing in ps aux) ───────────────────────
echo "→ writing .env to remote"
ssh "$PI_HOST" "echo 'export WANDB_API_KEY=$WANDB_API_KEY' > $REMOTE_DIR/.env"

# ── 4. launch in a persistent tmux session ───────────────────────────────────
TRAIN_CMD="python train.py configs/$DATASET.yaml device=cuda $OVERRIDES"

echo "→ launching training in tmux session '$SESSION'"
ssh "$PI_HOST" "
    tmux kill-session -t $SESSION 2>/dev/null || true
    tmux new-session -d -s $SESSION
    tmux send-keys -t $SESSION \
        'cd $REMOTE_DIR && source .env && $TRAIN_CMD 2>&1 | tee train.log' \
        Enter
"

echo ""
echo "  training is running — attach to watch:"
echo "  ssh $PI_HOST -t 'tmux attach -t $SESSION'"
echo ""
echo "  copy checkpoints when done:"
echo "  rsync -az $PI_HOST:$REMOTE_DIR/checkpoints/ ./checkpoints/"
