#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PYTHON="${LUMINA_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
UNILIP_PYTHON="${UNILIP_PYTHON:-/home/jiahao/miniconda3/envs/UniLIP/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/jiahao/task/UniLIP/data/csgo_benchmark_v2}"
SHARED_EVAL_DIR="${SHARED_EVAL_DIR:-/home/jiahao/task/csgo_benchmark_v2_eval_general}"
CONFIG="$PROJECT_ROOT/configs/csgo_seen10.json"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 {smoke|train|infer|eval} [--seed N] [--task all|discrete|continuous] [--resume [PATH]]" >&2
    exit 2
fi
MODE="$1"
shift
SEED="0"
TASK="all"
RESUME_PATH=""
CHECKPOINT_PATH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed)
            [[ $# -ge 2 ]] || { echo "--seed needs a value" >&2; exit 2; }
            SEED="$2"
            shift 2
            ;;
        --task)
            [[ $# -ge 2 ]] || { echo "--task needs a value" >&2; exit 2; }
            TASK="$2"
            shift 2
            ;;
        --resume)
            if [[ $# -ge 2 && "$2" != --* ]]; then
                RESUME_PATH="$2"
                shift 2
            else
                RESUME_PATH="auto"
                shift
            fi
            ;;
        --checkpoint)
            [[ $# -ge 2 ]] || { echo "--checkpoint needs a value" >&2; exit 2; }
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [[ ! -x "$MODEL_PYTHON" ]]; then
    echo "Lumina Python not found: $MODEL_PYTHON (run bash scripts/setup_csgo_seen10.sh first or set LUMINA_PYTHON)." >&2
    exit 2
fi

cd "$PROJECT_ROOT"
case "$MODE" in
    smoke)
        if [[ "$TASK" != "all" ]]; then
            echo "Smoke always checks one discrete and one continuous image; omit --task or use --task all." >&2
            exit 2
        fi
        "$MODEL_PYTHON" "$PROJECT_ROOT/train_seen10.py" --config "$CONFIG" --seed "$SEED" --smoke --resume auto
        "$MODEL_PYTHON" "$PROJECT_ROOT/infer_seen10.py" --config "$CONFIG" --seed "$SEED" --smoke --task all
        "$UNILIP_PYTHON" "$SHARED_EVAL_DIR/run_eval.py" smoke discrete \
            --config "$SHARED_EVAL_DIR/benchmark_v2.yaml" \
            --pred-root "$PROJECT_ROOT/outputs/csgo_benchmark_v2_seen10/Lumina-Image-2.0/seed_$SEED/smoke/discrete/gen_imgs" \
            --data-root "$DATA_ROOT" --limit 1
        "$UNILIP_PYTHON" "$SHARED_EVAL_DIR/run_eval.py" smoke continuous \
            --config "$SHARED_EVAL_DIR/benchmark_v2.yaml" \
            --pred-root "$PROJECT_ROOT/outputs/csgo_benchmark_v2_seen10/Lumina-Image-2.0/seed_$SEED/smoke/continuous/gen_imgs" \
            --data-root "$DATA_ROOT" --frame-only --max-clips 1
        ;;
    train)
        command=("$MODEL_PYTHON" "$PROJECT_ROOT/train_seen10.py" --config "$CONFIG" --seed "$SEED")
        if [[ -n "$RESUME_PATH" ]]; then
            command+=(--resume "$RESUME_PATH")
        fi
        "${command[@]}"
        ;;
    infer)
        command=("$MODEL_PYTHON" "$PROJECT_ROOT/infer_seen10.py" --config "$CONFIG" --seed "$SEED" --task "$TASK")
        if [[ -n "$CHECKPOINT_PATH" ]]; then
            command+=(--checkpoint "$CHECKPOINT_PATH")
        fi
        "${command[@]}"
        ;;
    eval)
        if [[ ! -x "$UNILIP_PYTHON" ]]; then
            echo "Shared evaluator Python not found: $UNILIP_PYTHON" >&2
            exit 2
        fi
        tasks=(discrete continuous)
        if [[ "$TASK" != "all" ]]; then
            tasks=("$TASK")
        fi
        for task in "${tasks[@]}"; do
            case "$task" in
                discrete|continuous) ;;
                *) echo "eval --task supports all, discrete or continuous (not $task)." >&2; exit 2 ;;
            esac
            "$UNILIP_PYTHON" "$SHARED_EVAL_DIR/run_eval.py" "$task" \
                --config "$SHARED_EVAL_DIR/benchmark_v2.yaml" \
                --pred-root "$PROJECT_ROOT/outputs/csgo_benchmark_v2_seen10/Lumina-Image-2.0/seed_$SEED/$task/gen_imgs" \
                --data-root "$DATA_ROOT" \
                --output "$PROJECT_ROOT/outputs/csgo_benchmark_v2_seen10/Lumina-Image-2.0/seed_$SEED/evaluation_shared/$task"
        done
        ;;
    *)
        echo "Unknown mode: $MODE" >&2
        exit 2
        ;;
esac
