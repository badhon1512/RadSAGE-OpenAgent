

unset SLURM_EXPORT_ENV
module load python

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80
export HTTP_PROXY=http://proxy.nhr.fau.de:80
export HTTPS_PROXY=http://proxy.nhr.fau.de:80
export PYTHONUNBUFFERED=1

# Required by bitsandbytes (libnvJitLink.so.13 ships inside the vllm conda env)
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/woody/iwi5/iwi5284h/software/private/conda/envs/rrg_vllm/lib/python3.11/site-packages/nvidia/cu13/lib"
# Disable FlashInfer JIT sampler — cluster has no nvcc; falls back to PyTorch-native sampling
export VLLM_USE_FLASHINFER_SAMPLER=0

# Persistent HF cache on /home/woody (large quota) — survives across jobs
export HF_HOME="/home/woody/iwi5/iwi5284h/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
mkdir -p "$HF_HUB_CACHE"

# HuggingFace token — required for gated models (e.g. Llama, Gemma)
ENV_FILE="/home/hpc/iwi5/iwi5284h/RRG/.env"
if [ -f "$ENV_FILE" ]; then
  export HF_TOKEN=$(grep -E "^HF_TOKEN=" "$ENV_FILE" | cut -d= -f2-)
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
  echo "HF token loaded from .env"
else
  echo "WARNING: .env not found — gated models will fail to download"
fi

# ── Resolve env paths (no conda activate needed in SLURM) ─────────────────────
# Use full binary path — .bashrc shell functions are not available in SLURM jobs
MAMBA_BIN="${HOME}/micromamba/bin/micromamba"

_env_path() {
  local name="$1"
  local p
  # Try micromamba binary first, then fall back to known base path
  p=$("$MAMBA_BIN" env list 2>/dev/null | awk -v n="$name" '$1==n{print $NF}')
  [ -z "$p" ] && p="/home/woody/iwi5/iwi5284h/software/private/conda/envs/$name"
  echo "$p"
}

RRG_ENV_PATH=$(_env_path "rrg_train")
VLLM_ENV_PATH=$(_env_path "rrg_vllm")
RRG_PYTHON="${RRG_ENV_PATH}/bin/python"
VLLM_PYTHON="${VLLM_ENV_PATH}/bin/python"

echo "RRG env:  $RRG_ENV_PATH"
echo "VLLM env: $VLLM_ENV_PATH"

if [ ! -f "$RRG_PYTHON" ]; then
  echo "ERROR: rrg_train python not found at $RRG_PYTHON"
  exit 1
fi

PROJECT_DIR="/home/hpc/iwi5/iwi5284h/RRG"
LOGDIR="$PROJECT_DIR/logs/orchestrator-agent-$SLURM_JOB_ID"
mkdir -p "$LOGDIR"
mkdir -p "$PROJECT_DIR/outputs3"

# ── Job parameters (all overridable via env vars) ─────────────────────────────
VERSION="v3"
MODEL_NAME="${MODEL_NAME:-Qwen3-4B}"
MAX_TOOL_CALLS="${MAX_TOOL_CALLS:-20}"
MAX_REVISION_ROUNDS="${MAX_REVISION_ROUNDS:-10}"
INPUT_CSV="${INPUT_CSV:-$PROJECT_DIR/srr_eval_all.csv}"
OUTPUT_CSV="${OUTPUT_CSV:-$PROJECT_DIR/outputs3/${MODEL_NAME}-orchestrator_agent-${VERSION}.csv}"
STATS_CSV="${STATS_CSV:-$PROJECT_DIR/outputs3/${MODEL_NAME}-orchestrator_agent_stat-${VERSION}.csv}"
OUTPUT_COLUMN="${OUTPUT_COLUMN:-${MODEL_NAME}-orchestrator_agent-${VERSION}}"
SELECT_FINAL="${SELECT_FINAL:-true}"
USE_VLLM="${USE_VLLM:-true}"
# Unique port per job — avoids collisions when multiple jobs run on same node
VLLM_PORT="${VLLM_PORT:-$((8100 + SLURM_JOB_ID % 900))}"
VLLM_GPU_MEM="${VLLM_GPU_MEM:-0.90}"
NUM_WORKERS="${NUM_WORKERS:-1}"

# ── Resolve full HuggingFace model ID (needed for vLLM server) ────────────────
case "$MODEL_NAME" in
  Qwen*)     FULL_MODEL_ID="Qwen/$MODEL_NAME" ;;
  gemma*)    FULL_MODEL_ID="google/$MODEL_NAME" ;;
  medgemma*) FULL_MODEL_ID="google/$MODEL_NAME" ;;
  Llama*)    FULL_MODEL_ID="meta-llama/$MODEL_NAME" ;;
  *)         FULL_MODEL_ID="$MODEL_NAME" ;;
esac

echo "========================================"
echo "Model:               $MODEL_NAME ($FULL_MODEL_ID)"
echo "Max tool calls:      $MAX_TOOL_CALLS"
echo "Max revision rounds: $MAX_REVISION_ROUNDS"
echo "Select final:        $SELECT_FINAL"
echo "Use vLLM:            $USE_VLLM"
echo "Num workers:         $NUM_WORKERS"
echo "Input CSV:           $INPUT_CSV"
echo "Output CSV:          $OUTPUT_CSV"
echo "Logs:                $LOGDIR"
echo "========================================"

# ── Build CLI flags ───────────────────────────────────────────────────────────
SELECT_FLAG="--select_final"
[ "$SELECT_FINAL" = "false" ] && SELECT_FLAG="--no-select_final"

OUTPUT_COLUMN_FLAG=""
[ -n "$OUTPUT_COLUMN" ] && OUTPUT_COLUMN_FLAG="--output_column $OUTPUT_COLUMN"

VLLM_FLAG=""
WORKER_FLAG=""
if [ "$USE_VLLM" = "true" ]; then
  VLLM_FLAG="--use_vllm"
  WORKER_FLAG="--num_workers $NUM_WORKERS"
fi

# ── Extract project to fast local scratch ─────────────────────────────────────
cd "$TMPDIR"
tar xzf "$WORK/rrg_project.tar.gz"

# Redirect vLLM/Triton cache to TMPDIR (job-isolated) — avoids cross-job race conditions when
# multiple jobs run on the same node and rm -rf ~/.cache/vllm deletes a shared directory mid-run
export VLLM_CACHE_ROOT="$TMPDIR/.cache/vllm"
export TRITON_CACHE_DIR="$TMPDIR/.cache/triton"
mkdir -p "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR"

# ── Start vLLM server (if requested) ─────────────────────────────────────────
VLLM_PID=""
if [ "$USE_VLLM" = "true" ]; then
  if [ ! -f "$VLLM_PYTHON" ]; then
    echo "ERROR: rrg_vllm environment not found at $VLLM_ENV_PATH"
    echo "Run:  sbatch job_setup_vllm.sh"
    exit 1
  fi

  echo "Starting vLLM server: $FULL_MODEL_ID on port $VLLM_PORT ..."
  NO_PROXY="127.0.0.1,localhost" \
  no_proxy="127.0.0.1,localhost" \
  "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \
    --model "$FULL_MODEL_ID" \
    --port "$VLLM_PORT" \
    --max-model-len 32768 \
    --gpu-memory-utilization "$VLLM_GPU_MEM" \
    --quantization bitsandbytes \
    --load-format bitsandbytes \
    --trust-remote-code \
    >> "$LOGDIR/vllm_server.out" 2>&1 &
  VLLM_PID=$!

  echo "vLLM PID=$VLLM_PID — sleeping 30s then polling for model ready..."
  sleep 30
  VLLM_READY=0
  for i in $(seq 1 120); do
    sleep 5
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
      echo "ERROR: vLLM process died during startup"
      tail -20 "$LOGDIR/vllm_server.out"
      exit 1
    fi
    # /v1/models only lists the model after it is fully loaded and ready to serve
    MODEL_LIST=$(curl --noproxy '*' -s "http://127.0.0.1:${VLLM_PORT}/v1/models" 2>/dev/null)
    if echo "$MODEL_LIST" | grep -q "$FULL_MODEL_ID"; then
      echo "vLLM model ready after $((i * 5))s — $FULL_MODEL_ID is serving"
      rm -rf ~/.cache/vllm 2>/dev/null; true
      VLLM_READY=1
      break
    fi
  done
  if [ "$VLLM_READY" -eq 0 ]; then
    echo "ERROR: vLLM model did not become ready within 10 minutes"
    tail -20 "$LOGDIR/vllm_server.out"
    kill "$VLLM_PID" 2>/dev/null
    exit 1
  fi

else
  rm -rf ~/.cache/vllm 2>/dev/null; true
fi

# ── Run pipeline ──────────────────────────────────────────────────────────────
echo "Starting orchestrator pipeline..."
"$RRG_PYTHON" orchestrator_agent/src/orchestrator.py \
  --input_csv  "$INPUT_CSV" \
  --output_csv "$OUTPUT_CSV" \
  --stats_csv  "$STATS_CSV" \
  --model_name "$MODEL_NAME" \
  --max_tool_calls "$MAX_TOOL_CALLS" \
  --max_revision_rounds "$MAX_REVISION_ROUNDS" \
  --openai_base_url "http://127.0.0.1:${VLLM_PORT}/v1" \
  --openai_model_name "$FULL_MODEL_ID" \
  $SELECT_FLAG \
  $VLLM_FLAG \
  $WORKER_FLAG \
  $OUTPUT_COLUMN_FLAG \
  --no-resume \
  > "$LOGDIR/${MODEL_NAME}-orchestrator_agent.out" 2>&1
STATUS=$?

# ── Stop vLLM server ──────────────────────────────────────────────────────────
if [ -n "$VLLM_PID" ]; then
  echo "Stopping vLLM server (PID $VLLM_PID)..."
  kill "$VLLM_PID" 2>/dev/null
  wait "$VLLM_PID" 2>/dev/null
fi

[ $STATUS -ne 0 ] && { echo "Pipeline failed (exit $STATUS)"; exit 1; }
echo "Pipeline done"
rm -rf ~/.cache/vllm 2>/dev/null; true

# ── Evaluation ────────────────────────────────────────────────────────────────
EVAL_ENV_PATH=$(_env_path "rad_eval")
"${EVAL_ENV_PATH}/bin/python" scripts/fed_eval.py --csv_path "$OUTPUT_CSV" \
  > "$LOGDIR/${MODEL_NAME}-orchestrator_agent-eval.out" 2>&1 || { echo "evaluation failed"; exit 1; }

echo "Job done"
rm -rf ~/.cache/vllm 2>/dev/null; true
