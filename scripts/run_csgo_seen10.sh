#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PYTHON="${LUMINA_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
UNILIP_PYTHON="${UNILIP_PYTHON:-/home/jiahao/miniconda3/envs/UniLIP/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/jiahao/task/UniLIP/data/csgo_benchmark_v2}"
SHARED_EVAL_DIR="${SHARED_EVAL_DIR:-/home/jiahao/task/csgo_benchmark_v2_eval_general}"
CONFIG="$PROJECT_ROOT/configs/csgo_seen10.json"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 {smoke|train|infer|eval} [--seed N] [--task all|discrete|continuous] [--resume [PATH]] [--inference-engine eager|compiled] [--compile-mode reduce-overhead|default] [--batch-size N] [--vae-batch-size N] [--benchmark-batches N] [--output-root PATH]" >&2
    exit 2
fi
MODE="$1"
shift
SEED="0"
TASK="all"
RESUME_PATH=""
CHECKPOINT_PATH=""
ENGINE="eager"
COMPILE_MODE=""
BATCH_SIZE="1"
VAE_BATCH_SIZE="1"
BENCHMARK_BATCHES=""
OUTPUT_ROOT=""
JPEG_WORKERS="2"
JPEG_QUEUE_SIZE="8"

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
        --inference-engine|--engine)
            [[ $# -ge 2 ]] || { echo "--inference-engine needs a value" >&2; exit 2; }
            ENGINE="$2"
            shift 2
            ;;
        --compile-mode)
            [[ $# -ge 2 ]] || { echo "--compile-mode needs a value" >&2; exit 2; }
            COMPILE_MODE="$2"
            shift 2
            ;;
        --batch-size)
            [[ $# -ge 2 ]] || { echo "--batch-size needs a value" >&2; exit 2; }
            BATCH_SIZE="$2"
            shift 2
            ;;
        --vae-batch-size)
            [[ $# -ge 2 ]] || { echo "--vae-batch-size needs a value" >&2; exit 2; }
            VAE_BATCH_SIZE="$2"
            shift 2
            ;;
        --benchmark-batches)
            [[ $# -ge 2 ]] || { echo "--benchmark-batches needs a value" >&2; exit 2; }
            BENCHMARK_BATCHES="$2"
            shift 2
            ;;
        --output-root)
            [[ $# -ge 2 ]] || { echo "--output-root needs a value" >&2; exit 2; }
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --jpeg-workers)
            [[ $# -ge 2 ]] || { echo "--jpeg-workers needs a value" >&2; exit 2; }
            JPEG_WORKERS="$2"
            shift 2
            ;;
        --jpeg-queue-size)
            [[ $# -ge 2 ]] || { echo "--jpeg-queue-size needs a value" >&2; exit 2; }
            JPEG_QUEUE_SIZE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [[ -n "$COMPILE_MODE" ]]; then
    case "$COMPILE_MODE" in
        reduce-overhead|default) ;;
        *) echo "--compile-mode must be reduce-overhead or default." >&2; exit 2 ;;
    esac
    if [[ "$ENGINE" != "compiled" ]]; then
        echo "--compile-mode requires --inference-engine compiled." >&2
        exit 2
    fi
fi
if [[ -n "$BENCHMARK_BATCHES" ]]; then
    if [[ ! "$BENCHMARK_BATCHES" =~ ^[1-9][0-9]*$ ]]; then
        echo "--benchmark-batches must be a positive integer." >&2
        exit 2
    fi
    if [[ "$MODE" != "infer" ]]; then
        echo "--benchmark-batches is available only with infer." >&2
        exit 2
    fi
    if [[ "$TASK" == "all" ]]; then
        echo "Benchmark one task at a time with --task discrete or --task continuous." >&2
        exit 2
    fi
    if [[ -z "$OUTPUT_ROOT" ]]; then
        echo "--benchmark-batches requires an explicit independent --output-root." >&2
        exit 2
    fi
fi

if [[ ! -x "$MODEL_PYTHON" ]]; then
    echo "Lumina Python not found: $MODEL_PYTHON (run bash scripts/setup_csgo_seen10.sh first or set LUMINA_PYTHON)." >&2
    exit 2
fi

PREDICTION_BASE="$PROJECT_ROOT/outputs/csgo_benchmark_v2_seen10/Lumina-Image-2.0"
if [[ -n "$OUTPUT_ROOT" ]]; then
    PREDICTION_BASE="$OUTPUT_ROOT"
fi
INFER_OPTIONS=(--inference-engine "$ENGINE" --batch-size "$BATCH_SIZE" --vae-batch-size "$VAE_BATCH_SIZE"
    --jpeg-workers "$JPEG_WORKERS" --jpeg-queue-size "$JPEG_QUEUE_SIZE")
if [[ -n "$COMPILE_MODE" ]]; then
    INFER_OPTIONS+=(--compile-mode "$COMPILE_MODE")
fi
if [[ -n "$BENCHMARK_BATCHES" ]]; then
    INFER_OPTIONS+=(--benchmark-batches "$BENCHMARK_BATCHES")
fi

cd "$PROJECT_ROOT"
case "$MODE" in
    smoke)
        if [[ "$TASK" != "all" ]]; then
            echo "Smoke always checks one discrete and one continuous image; omit --task or use --task all." >&2
            exit 2
        fi
        "$MODEL_PYTHON" "$PROJECT_ROOT/train_seen10.py" --config "$CONFIG" --seed "$SEED" --smoke --resume auto
        infer_command=("$MODEL_PYTHON" "$PROJECT_ROOT/infer_seen10.py" --config "$CONFIG" --seed "$SEED" --smoke --task all
            "${INFER_OPTIONS[@]}")
        if [[ -n "$OUTPUT_ROOT" ]]; then
            infer_command+=(--output-root "$OUTPUT_ROOT")
        fi
        "${infer_command[@]}"
        "$UNILIP_PYTHON" "$SHARED_EVAL_DIR/run_eval.py" smoke discrete \
            --config "$SHARED_EVAL_DIR/benchmark_v2.yaml" \
            --pred-root "$PREDICTION_BASE/seed_$SEED/smoke/discrete/gen_imgs" \
            --data-root "$DATA_ROOT" --limit 1
        "$UNILIP_PYTHON" "$SHARED_EVAL_DIR/run_eval.py" smoke continuous \
            --config "$SHARED_EVAL_DIR/benchmark_v2.yaml" \
            --pred-root "$PREDICTION_BASE/seed_$SEED/smoke/continuous/gen_imgs" \
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
        command=("$MODEL_PYTHON" "$PROJECT_ROOT/infer_seen10.py" --config "$CONFIG" --seed "$SEED" --task "$TASK"
            "${INFER_OPTIONS[@]}")
        if [[ -n "$CHECKPOINT_PATH" ]]; then
            command+=(--checkpoint "$CHECKPOINT_PATH")
        fi
        if [[ -n "$OUTPUT_ROOT" ]]; then
            command+=(--output-root "$OUTPUT_ROOT")
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
                --pred-root "$PREDICTION_BASE/seed_$SEED/$task/gen_imgs" \
                --data-root "$DATA_ROOT" \
                --output "$PREDICTION_BASE/seed_$SEED/evaluation_shared/$task"
        done
        ;;
    *)
        echo "Unknown mode: $MODE" >&2
        exit 2
        ;;
esac
