#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"
BASE_MODEL_DIR="$PROJECT_ROOT/checkpoints/Lumina-Image-2.0"
LUMINA_MODEL_REVISION="53504abd8178b30685b6c4c7a4cd181ff78b73e9"
CLONE_ENV="${LUMINA_CLONE_ENV:-/home/jiahao/task/ControlAR/.venv}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

exec 9>"$PROJECT_ROOT/.setup_csgo_seen10.lock"
if command -v flock >/dev/null 2>&1 && ! flock -n 9; then
    echo "Another Seen-10 environment setup is already running." >&2
    exit 2
fi

if [[ -e "$VENV_DIR" && ! -x "$VENV_DIR/bin/python" ]]; then
    # Preserve an interrupted clone for inspection while clearing the exact
    # conda prefix, which must be absent before a safe retry.
    INCOMPLETE_VENV="${VENV_DIR}.incomplete.$(date +%Y%m%dT%H%M%S).$$"
    mv "$VENV_DIR" "$INCOMPLETE_VENV"
    echo "Moved incomplete environment to $INCOMPLETE_VENV"
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    if [[ -x "$CLONE_ENV/bin/python" ]] && command -v conda >/dev/null 2>&1; then
        # This host already has a Blackwell-tested torch/FlashAttention pair.
        conda create -y -p "$VENV_DIR" --clone "$CLONE_ENV"
    else
        if [[ -n "${LUMINA_SETUP_PYTHON:-}" ]]; then
            SETUP_PYTHON="$LUMINA_SETUP_PYTHON"
        elif command -v python3.11 >/dev/null 2>&1; then
            SETUP_PYTHON="$(command -v python3.11)"
        else
            echo "Set LUMINA_CLONE_ENV to a compatible environment or LUMINA_SETUP_PYTHON to Python 3.11." >&2
            exit 2
        fi
        "$SETUP_PYTHON" -m venv "$VENV_DIR"
        "$VENV_DIR/bin/python" -m pip install --upgrade pip
        "$VENV_DIR/bin/python" -m pip install --index-url "$TORCH_INDEX_URL" torch==2.8.0 torchvision==0.23.0
        "$VENV_DIR/bin/python" -m pip install flash-attn==2.8.3 --no-build-isolation
    fi
fi

LUMINA_PYTHON="$VENV_DIR/bin/python"
"$LUMINA_PYTHON" -m pip install --upgrade pip
"$LUMINA_PYTHON" -m pip install -r "$PROJECT_ROOT/requirements_seen10.txt"
"$LUMINA_PYTHON" - <<'PY'
import flash_attn
import torch

if not torch.cuda.is_available():
    raise RuntimeError("The Lumina environment cannot see a CUDA GPU")
capability = torch.cuda.get_device_capability()
if capability >= (12, 0) and "sm_120" not in torch.cuda.get_arch_list():
    raise RuntimeError(f"PyTorch {torch.__version__} does not contain sm_120 kernels for this Blackwell GPU")
print(f"validated torch={torch.__version__}, flash_attn={flash_attn.__version__}, capability={capability}")
PY

mkdir -p "$BASE_MODEL_DIR"
LUMINA_BASE_MODEL_DIR="$BASE_MODEL_DIR" \
LUMINA_MODEL_REVISION="$LUMINA_MODEL_REVISION" \
HF_TOKEN="${HF_TOKEN:-}" \
"$LUMINA_PYTHON" -c 'import os; from huggingface_hub import snapshot_download; snapshot_download(repo_id="Alpha-VLLM/Lumina-Image-2.0", revision=os.environ["LUMINA_MODEL_REVISION"], local_dir=os.environ["LUMINA_BASE_MODEL_DIR"], allow_patterns=["consolidated.00-of-01.pth", "model_args.pth", "vae/**"], token=os.environ.get("HF_TOKEN") or None)'

(
    cd "$BASE_MODEL_DIR"
    printf '%s  %s\n' \
        'a7c09ebae62996a8289782161338a3cdba58c11d2d849c50b2d6502e152b0d6d' \
        'consolidated.00-of-01.pth' \
        '291ae050720fcc5d2ed6d51a7d4b2f80f83972e2469e9b0362fce490b2fd1d6b' \
        'model_args.pth' \
        '80c5ed836c7171f6817a8addb910cdd32d30ca65549ba13b75e097c0c30fb326' \
        'vae/config.json' \
        '8c717328c8ad41faab2ccfd52ae17332505c6833cf176aad56e7b58f2c4d4c94' \
        'vae/diffusion_pytorch_model.safetensors' | sha256sum --check --strict
)

echo "Lumina environment: $LUMINA_PYTHON"
echo "Base NextDiT and FLUX VAE: $BASE_MODEL_DIR"
