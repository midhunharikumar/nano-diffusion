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
#   MAX_RUNTIME=3600 DATASET=imagenet256 ./launchers/prime_intellect.sh
#   PREPARE_DATA=1 DATASET=imagenet64 ./launchers/prime_intellect.sh  # pre-cache dataset before training

set -euo pipefail

# ── config ───────────────────────────────────────────────────────────────────
UV_VENV_CLEAR="${UV_VENV_CLEAR:-1}"
PI_HOST="${PI_HOST:-ubuntu@204.52.29.118}"
REMOTE_DIR="/ephemeral/nano-diffusion"    # fast NVMe scratch — wiped on instance termination
REMOTE_CACHE="/ephemeral/.cache"          # all package/model caches live on NVMe, not the small root disk
DATASET="${DATASET:-imagenet64}"
OVERRIDES="${OVERRIDES:-}"
MAX_RUNTIME="${MAX_RUNTIME:-}"            # optional: stop after N seconds
WANDB_API_KEY="${WANDB_API_KEY:-}"
HF_TOKEN="${HF_TOKEN:-}"                 # required for gated datasets (e.g. ImageNet)
GCP_BUCKET="${GCP_BUCKET:-}"             # GCS bucket for checkpoint uploads (e.g. "checkpoints")
GCP_CREDENTIALS_FILE="${GCP_CREDENTIALS_FILE:-}"  # path to service account JSON on local machine
CUDA_TAG="${CUDA_TAG:-cu126}"            # cu126 (driver>=560), cu124 (driver>=530), cu121 (driver>=525), cu118 (driver>=450)
PREPARE_DATA="${PREPARE_DATA:-0}"        # set to 1 to pre-download dataset to NVMe before training
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
if [[ -z "$HF_TOKEN" ]]; then
    echo "warning: HF_TOKEN is not set — gated datasets (e.g. ImageNet) will fail"
fi
if [[ -n "$GCP_BUCKET" && -z "$GCP_CREDENTIALS_FILE" ]]; then
    echo "warning: GCP_BUCKET set but GCP_CREDENTIALS_FILE is empty — using ADC (may fail)"
fi

# ── 1. sync source + GCP credentials ─────────────────────────────────────────
echo "→ syncing code to $PI_HOST:$REMOTE_DIR"
ssh "$PI_HOST" "mkdir -p $REMOTE_DIR"
rsync -az --progress \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='checkpoints' \
    --exclude='launchers' \
    --exclude='*.pt' \
    "$ROOT/" "$PI_HOST:$REMOTE_DIR/"

# copy GCP service account key if provided — store in home dir, not /ephemeral,
# so it survives across re-deploys on the same instance
if [[ -n "$GCP_CREDENTIALS_FILE" ]]; then
    echo "→ copying GCP credentials"
    ssh "$PI_HOST" "mkdir -p ~/.gcp"
    scp "$GCP_CREDENTIALS_FILE" "$PI_HOST:~/.gcp/credentials.json"
fi

# ── 2. install uv + create venv (idempotent) ─────────────────────────────────
echo "→ setting up uv environment"
ssh "$PI_HOST" "
    set -e
    export PATH=\"\$HOME/.local/bin:\$PATH\"

    # keep all package caches on NVMe — the root disk is small and fills up fast
    mkdir -p $REMOTE_CACHE/uv $REMOTE_CACHE/pip
    export XDG_CACHE_HOME=$REMOTE_CACHE
    export UV_CACHE_DIR=$REMOTE_CACHE/uv
    export PIP_CACHE_DIR=$REMOTE_CACHE/pip

    # install uv if not present
    if ! command -v uv &>/dev/null; then
        mkdir -p \"\$HOME/.config/uv\" 2>/dev/null || true
        curl -LsSf https://astral.sh/uv/install.sh | sh || true
    fi
    # confirm binary is available after install
    command -v uv

    cd $REMOTE_DIR

    # create .venv; --seed includes pip so we can use it for the PyTorch wheel server
    UV_VENV_CLEAR=$UV_VENV_CLEAR uv venv --quiet --seed

    # Step 1: torch/torchvision via pip — pip handles PyTorch's non-standard wheel
    # server natively; uv's --index-url can't resolve it.
    .venv/bin/pip install --quiet \
        --index-url "https://download.pytorch.org/whl/$CUDA_TAG" \
        torch torchvision

    # Step 2: everything else via uv (faster); skip torch lines so uv never
    # tries to re-resolve them against PyPI.
    grep -vE '^torch(vision)?(==|>=|<=|~=|!=| |$)' requirements.txt > $REMOTE_DIR/.reqs_no_torch.txt
    uv pip install --quiet -r $REMOTE_DIR/.reqs_no_torch.txt
"

# ── 3. write env file (avoids key appearing in ps aux) ───────────────────────
echo "→ writing .env to remote"
ssh "$PI_HOST" "
    echo 'export WANDB_API_KEY=$WANDB_API_KEY'                                           > $REMOTE_DIR/.env
    echo 'export HF_TOKEN=$HF_TOKEN'                                                    >> $REMOTE_DIR/.env
    echo 'export HF_XET_HIGH_PERFORMANCE=1'                                             >> $REMOTE_DIR/.env
    echo 'export HF_DATASETS_CACHE=$REMOTE_CACHE/huggingface/datasets'                  >> $REMOTE_DIR/.env
    echo 'export HF_HOME=$REMOTE_CACHE/huggingface'                                     >> $REMOTE_DIR/.env
    # Cosmos tokenizer weights (tokenizer.py reads COSMOS_CACHE_DIR) + torch/triton
    # compile caches + uv/pip — keep every cache on NVMe, never the small root disk.
    echo 'export COSMOS_CACHE_DIR=$REMOTE_CACHE/huggingface/cosmos'                      >> $REMOTE_DIR/.env
    echo 'export XDG_CACHE_HOME=$REMOTE_CACHE'                                           >> $REMOTE_DIR/.env
    echo 'export UV_CACHE_DIR=$REMOTE_CACHE/uv'                                          >> $REMOTE_DIR/.env
    echo 'export PIP_CACHE_DIR=$REMOTE_CACHE/pip'                                        >> $REMOTE_DIR/.env
    echo 'export TORCH_HOME=$REMOTE_CACHE/torch'                                         >> $REMOTE_DIR/.env
    echo 'export TRITON_CACHE_DIR=$REMOTE_CACHE/triton'                                  >> $REMOTE_DIR/.env
    [[ -n '$GCP_CREDENTIALS_FILE' ]] && \
        echo 'export GOOGLE_APPLICATION_CREDENTIALS=~/.gcp/credentials.json'            >> $REMOTE_DIR/.env
"

# ── 4. (optional) pre-download dataset to local NVMe cache ───────────────────
if [[ "$PREPARE_DATA" == "1" ]]; then
    echo "→ pre-downloading $DATASET to local cache (this may take a few minutes)"
    ssh "$PI_HOST" "
        export PATH=\"\$HOME/.local/bin:\$PATH\"
        cd $REMOTE_DIR
        source .env
        uv run python prepare_data.py $DATASET
    "
fi

# ── 5. launch in a persistent tmux session ───────────────────────────────────
TRAIN_CMD="uv run python train.py configs/$DATASET.yaml device=cuda $OVERRIDES"
[[ -n "$MAX_RUNTIME" ]] && TRAIN_CMD="$TRAIN_CMD --max_runtime $MAX_RUNTIME"
[[ -n "$GCP_BUCKET"  ]] && TRAIN_CMD="$TRAIN_CMD --gcp_bucket $GCP_BUCKET"

echo "→ launching training in tmux session '$SESSION'"
ssh "$PI_HOST" "
    export PATH=\"\$HOME/.local/bin:\$PATH\"
    tmux kill-session -t $SESSION 2>/dev/null || true
    tmux new-session -d -s $SESSION
    tmux send-keys -t $SESSION \
        'export PATH=\"\$HOME/.local/bin:\$PATH\" && cd $REMOTE_DIR && source .env && $TRAIN_CMD 2>&1 | tee train.log' \
        Enter
"

echo ""
echo "  training is running — attach to watch:"
echo "  ssh $PI_HOST -t 'tmux attach -t $SESSION'"
echo ""
echo "  copy checkpoints when done:"
echo "  rsync -az $PI_HOST:$REMOTE_DIR/checkpoints/ ./checkpoints/"
