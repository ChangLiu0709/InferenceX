#!/usr/bin/env bash

# Agentic trace-replay recipe for a disaggregated SGLang server on MI355X
# (GLM-5.2 MXFP4, 1P1D TP8/EP8 with MoRI KV transfer).
#
# Driven by environment variables; submits a SLURM job via submit.sh.
# GLM-5.2 uses DSA/tilelang attention backends and fp8_e4m3 KV cache.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../benchmark_lib.sh"

check_env_vars \
    CONC_LIST \
    ISL \
    OSL \
    IMAGE \
    SPEC_DECODING \
    MODEL_PATH \
    PREFILL_NUM_WORKERS \
    PREFILL_TP \
    PREFILL_EP \
    PREFILL_DP_ATTN \
    DECODE_NUM_WORKERS \
    DECODE_TP \
    DECODE_EP \
    DECODE_DP_ATTN \
    PREFILL_NODES \
    DECODE_NODES \
    RANDOM_RANGE_RATIO \
    DURATION \
    KV_OFFLOADING \
    IS_AGENTIC \
    FRAMEWORK

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

set -x

# Use upstreamed multi_node scripts (no external clone needed)
cd "$GITHUB_WORKSPACE/benchmarks/multi_node/amd_utils" || exit 1

# Set up SGL launch script-specific environment variables
export TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
export MODEL_PATH=$MODEL_PATH
export MODEL_NAME=$MODEL_NAME
export CONTAINER_IMAGE=$IMAGE

# ── Identity / result naming ──
export MODEL_PREFIX="${MODEL_PREFIX:-glm5.2}"
export PRECISION="${PRECISION:-fp4}"
export RESULT_FILENAME="${RESULT_FILENAME:-${RUNNER_NAME:-glm5.2-fp4-agentic}}"

# ── Agentic benchmark params ──
export DURATION="${DURATION:-1800}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-1000000}"

# ── In-tree sglang patches ──
# GLM-5.2 uses hybrid state (DSA); the mori_conn.py patch targets this.
export MORI_CONN_PATCH="${MORI_CONN_PATCH:-apply}"

# ── Aiter fault mitigation ──
export DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-0}"

# ── KV cache offloading (HiCache) ──
export KV_OFFLOADING="${KV_OFFLOADING:-none}"
if [[ "$KV_OFFLOADING" != "none" ]]; then
  export KV_OFFLOAD_BACKEND="${KV_OFFLOAD_BACKEND:-hicache}"
fi
if [[ "$KV_OFFLOADING" != "none" && "${KV_OFFLOAD_BACKEND:-}" == "hicache" ]]; then
  export HICACHE_TIER="${HICACHE_TIER:-L2}"
  export HICACHE_HOST_POOL_COUNT="${HICACHE_HOST_POOL_COUNT:-1}"
  export HICACHE_PAGE_SIZE="${HICACHE_PAGE_SIZE:-}"
  export HICACHE_RATIO="${HICACHE_RATIO:-}"

  if [[ "${HICACHE_TIER^^}" == "L3" ]]; then
    export HICACHE_MEM_LAYOUT="${HICACHE_MEM_LAYOUT:-page_first}"
    export HICACHE_IO_BACKEND="${HICACHE_IO_BACKEND:-direct}"
    export HICACHE_WRITE_POLICY="${HICACHE_WRITE_POLICY:-write_through}"
    export HICACHE_STORAGE_BACKEND="${HICACHE_STORAGE_BACKEND:-mooncake}"
  else
    export HICACHE_MEM_LAYOUT="${HICACHE_MEM_LAYOUT:-page_first}"
    export HICACHE_IO_BACKEND="${HICACHE_IO_BACKEND:-direct}"
    export HICACHE_WRITE_POLICY="${HICACHE_WRITE_POLICY:-write_through}"
    export HICACHE_STORAGE_BACKEND="${HICACHE_STORAGE_BACKEND:-}"
  fi
  export HICACHE_PREFETCH_POLICY="${HICACHE_PREFETCH_POLICY:-best_effort}"
  export MC_MASTER_PORT="${MC_MASTER_PORT:-58137}"
  export MC_METADATA_PORT="${MC_METADATA_PORT:-8080}"
  export MC_METRICS_PORT="${MC_METRICS_PORT:-19003}"
  export MC_MASTER_THREADS="${MC_MASTER_THREADS:-64}"
  export MC_EVICTION_HIGH_WATERMARK="${MC_EVICTION_HIGH_WATERMARK:-0.95}"
  export MC_PATCH_HOSTPOOL="${MC_PATCH_HOSTPOOL:-1}"
  export MC_PROTOCOL="${MC_PROTOCOL:-tcp}"
  export MC_GLOBAL_SEG="${MC_GLOBAL_SEG:-64gb}"
  export MC_DEVICE="${MC_DEVICE:-}"
  export MC_MASTER_ADDR="${MC_MASTER_ADDR:-}"
  export MC_METADATA_SERVER="${MC_METADATA_SERVER:-}"
fi

# ── MoRIIO RDMA Send Queue tuning ──
export MORI_IO_SQ_BACKOFF_TIMEOUT_US="${MORI_IO_SQ_BACKOFF_TIMEOUT_US:-500000}"
export MORI_IO_QP_MAX_SEND_WR="${MORI_IO_QP_MAX_SEND_WR:-32768}"

# ── SGLang PD router policy + server metrics ──
export PREFILL_ROUTER_POLICY="${PREFILL_ROUTER_POLICY:-cache_aware}"
export ENABLE_METRICS="${ENABLE_METRICS:-1}"

# ── MTP ──
export DECODE_MTP_SIZE="${DECODE_MTP_SIZE:-0}"

# Derive EP/DP enable flags from the topology inputs.
if [[ "${PREFILL_EP:-1}" -eq 1 ]]; then
export PREFILL_ENABLE_EP=false
else
export PREFILL_ENABLE_EP=true
fi

if [[ "$PREFILL_DP_ATTN" == "true" ]]; then
export PREFILL_ENABLE_DP=true
else
export PREFILL_ENABLE_DP=false
fi

if [[ "${DECODE_EP:-1}" -eq 1 ]]; then
export DECODE_ENABLE_EP=false
else
export DECODE_ENABLE_EP=true
fi

if [[ "$DECODE_DP_ATTN" == "true" ]]; then
export DECODE_ENABLE_DP=true
else
export DECODE_ENABLE_DP=false
fi

# Launch the job. CONC_LIST is space-delimited in YAML; submit.sh wants 'x'.
JOB_ID=$(bash ./submit.sh $PREFILL_NODES \
    $PREFILL_NUM_WORKERS \
    $DECODE_NODES \
    $DECODE_NUM_WORKERS \
    $ISL $OSL "${CONC_LIST// /x}" inf \
    ${PREFILL_ENABLE_EP} ${PREFILL_ENABLE_DP} \
    ${DECODE_ENABLE_EP} ${DECODE_ENABLE_DP} \
    ${PREFILL_TP} ${DECODE_TP} \
    ${RANDOM_RANGE_RATIO})

if [[ $? -ne 0 ]]; then
    echo "Failed to submit job" >&2
    exit 1
fi

echo "$JOB_ID"
