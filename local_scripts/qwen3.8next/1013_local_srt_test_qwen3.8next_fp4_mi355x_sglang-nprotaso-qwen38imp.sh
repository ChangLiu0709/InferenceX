#!/usr/bin/env bash
# =============================================================================
# Local smoke test: Qwen3.8-Flash-Next MXFP4 (Quark, FP8 PLE) AgentX on MI355X
# via SGLang with native NEXTN MTP, launched through the new InferenceX
# srt-slurm flow (the same `python -m infx.launch run` CI uses).
#
# CI config (inferencex-e2e/configs/amd-master.yaml):
#   qwen3.8next-fp4-mi355x-sglang-agentic-mtp
#     image:  lmsysorg/sglang:v0.5.21-rocm724-mi35x@sha256:2f6b9105e26f...
#     model:  amd/Qwen3.8-Flash-Next-Quark-MXFP4-PLEFP8, model-prefix qwen3.8next
#     runner: cluster:mi355x-amds, precision fp4, framework sglang
#     agentic-coding, dram-utilization 0.80
#     tp1 ep1 mtp, kv-offloading none, conc-list [1, 4, 8, 12, 16]
#     srt-recipe: benchmarks/single_node/srt-slurm-recipes/qwen3.8next/sglang/mi355x-fp4-mtp/agentic.yaml
#
# Differences from CI, all local to this script:
#   - IMAGE defaults to aigmkt/sglang:nprotaso_qwen3_8_improvement (Nikolai's
#     build of amd-nprotaso qwen_3_8_improvement: AITER badcd62a, SGLang
#     92e3addcd1). The launcher binds the recipe's model.container from IMAGE.
#     The feature flags below are added to the server env (roles.agg.env) of a
#     private copy of the recipe.
#   - A private runners.yaml: unique runner name (launcher cleanup scancels by
#     runner name, so never reuse a CI runner), squash dir and srt-slurm outputs
#     we own, MODEL_HOST_DIR staged as the config's MODEL, and a host-setup hook
#     that keeps CI's GPU drain gate but drops its sudo hugepage steps.
#   - Each concurrency runs in its own copy of inferencex-e2e (like CI's fresh
#     checkout), so the InferenceX branch stays clean.
#
# Matrix points (TP/EP, exp-name, KV offload, CPU DRAM, duration) come from
# `infx.matrix.generate` for CONFIG_KEY, so they track amd-master.yaml.
# Golden acceptance length (SGLANG_SIMULATE_ACC_LEN) is set by the launcher from
# MODEL_PREFIX + THINKING_MODE (qwen3.8next, thinking_on, 3 steps -> 2.32).
# AgentX dataset: the launcher default for 256k-capped families
# (semianalysis_cc_traces_weka_062126_256k), cached in /it-share/hf-hub-cache.
#
# Usage:
#   bash 1013_local_srt_test_qwen3.8next_fp4_mi355x_sglang-nprotaso-qwen38imp.sh
#   CONC_LIST="1 4" bash 1013_local_srt_test_qwen3.8next_fp4_mi355x_sglang-nprotaso-qwen38imp.sh
#   AITER_HC_MIX=0 DURATION=900 bash 1013_local_srt_test_qwen3.8next_fp4_mi355x_sglang-nprotaso-qwen38imp.sh
#   DRY_RUN=1 bash ...   # prepare and bind every point, submit nothing
#   SCRATCH=/it-share/<you> bash ...   # base for InferenceX, venv, squash, outputs, logs
# =============================================================================
set -euo pipefail

SCRATCH="${SCRATCH:-/it-share/${USER%%@*}}"
[[ -d "$SCRATCH" ]] || { echo "ERROR: scratch dir not found: $SCRATCH (set SCRATCH=/it-share/<you>)" >&2; exit 1; }
INFERENCEX_DIR="${INFERENCEX_DIR:-${SCRATCH}/InferenceX}"
E2E_DIR="${INFERENCEX_DIR}/inferencex-e2e"

# ---- CI config entry and image ----
CONFIG_FILE="${CONFIG_FILE:-configs/amd-master.yaml}"
CONFIG_KEY="${CONFIG_KEY:-qwen3.8next-fp4-mi355x-sglang-agentic-mtp}"
IMAGE="${IMAGE:-aigmkt/sglang:nprotaso_qwen3_8_improvement}"

# ---- Points to run (subset of the config's conc-list) ----
CONC_LIST="${CONC_LIST:-1}"
# Empty keeps the config's duration (3600 s).
DURATION="${DURATION:-}"
THINKING_MODE="${THINKING_MODE:-thinking_on}"

# ---- Nikolai feature flags (server env) ----
EXACT_CHUNK_FILL="${EXACT_CHUNK_FILL:-1}"
AITER_GDN_CONV="${AITER_GDN_CONV:-1}"
AITER_GDN_DECODE="${AITER_GDN_DECODE:-1}"
AITER_HC_MIX="${AITER_HC_MIX:-1}"
AITER_QSA_PA_DECODE="${AITER_QSA_PA_DECODE:-1}"
AITER_GDR_FLYDSL="${AITER_GDR_FLYDSL:-1}"
AITER_TOPK_GATING="${AITER_TOPK_GATING:-1}"
AITER_GDN_PREFILL="${AITER_GDN_PREFILL:-1}"
AITER_GDN_PREFILL_H="${AITER_GDN_PREFILL_H:-fused}"
# Boolean: 1 uses --mem-fraction-static as given; 0 lets SGLang scale it by 0.85
# for context > 8192 (attention_backend=aiter). Non-boolean values fall back to 0.
AITER_HONOR_EXPLICIT_MEM_FRACTION="${AITER_HONOR_EXPLICIT_MEM_FRACTION:-0}"

# ---- Launcher environment ----
LAUNCH_PYTHON="${LAUNCH_PYTHON:-${SCRATCH}/venvs/infx-launch-py312/bin/python}"
export PATH="${HOME}/.local/bin:${PATH}"   # uv, for srtctl's venv
# Unique per user: launcher cleanup scancels every job named after the runner.
RUNNER_NAME="${RUNNER_NAME:-$(echo "${USER%%@*}" | tr -c 'a-zA-Z0-9\n' '-')-mi355x-smoke_00}"
SQUASH_DIR="${SQUASH_DIR:-${SCRATCH}/squash}"
SRT_OUTPUTS="${SRT_OUTPUTS:-${SCRATCH}/srt-slurm/outputs}"
HF_HUB_CACHE_HOST="${HF_HUB_CACHE_HOST:-/it-share/hf-hub-cache}"
# Weights served for the config's MODEL: the fanwu103 MXFP4-PLEFP8 weights every
# earlier run used. The PLE table loads in BF16 (~95 GiB) before its FP8 swap, so
# a GPU with leaked VRAM OOMs at load; the host-setup hook waits for clean GPUs.
MODEL_HOST_DIR="${MODEL_HOST_DIR:-/it-share/hf-hub-cache/Qwen3.8-Flash-Next-MXFP4-PLEFP8}"
SLURM_EXCLUDE_NODES="${SLURM_EXCLUDE_NODES:-}"
DRY_RUN="${DRY_RUN:-0}"

# ---- Output ----
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-${SCRATCH}/logs/qwen3.8-flash-next}"
RUN_DIR="${RUN_DIR:-${LOG_DIR}/srt_sglang_fp4_mtp_agentx_${TIMESTAMP}_nprotaso_qwen38imp}"
mkdir -p "$RUN_DIR" "$SRT_OUTPUTS"

SQUASH_FILE="${SQUASH_DIR}/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"

# ---- Verify prerequisites ----
fail() { echo "ERROR: $*" >&2; exit 1; }
[[ -f "$E2E_DIR/$CONFIG_FILE" ]] || fail "config not found: $E2E_DIR/$CONFIG_FILE"
[[ -e "$E2E_DIR/utils/srt-slurm/.git" ]] \
    || fail "srt-slurm submodule missing; run: git -C $INFERENCEX_DIR submodule update --init inferencex-e2e/utils/srt-slurm"
[[ -x "$LAUNCH_PYTHON" ]] || fail "launcher Python 3.12 venv not found: $LAUNCH_PYTHON"
"$LAUNCH_PYTHON" -c 'import pydantic, yaml, srtctl' 2>/dev/null \
    || fail "launcher venv needs infx deps and srtctl (uv pip install --python $LAUNCH_PYTHON -r $E2E_DIR/pyproject.toml $E2E_DIR/utils/srt-slurm)"
command -v uv >/dev/null || fail "uv not on PATH (python3 -m pip install --user uv)"
[[ -f "$SQUASH_FILE" ]] || echo "NOTE: $SQUASH_FILE not found; the launcher imports $IMAGE on the compute node"
if [[ "$DRY_RUN" != 1 ]] && ! timeout 8 sinfo -N --noheader >/dev/null 2>&1; then
    fail "Slurm is not reachable from this host"
fi

# ---- Matrix points for CONFIG_KEY ----
POINTS_JSON="$RUN_DIR/matrix_points.json"
(cd "$E2E_DIR" && PYTHONPATH="$E2E_DIR" "$LAUNCH_PYTHON" -P -m infx.matrix.generate test-config \
    --config-files "$CONFIG_FILE" --no-evals --config-keys "$CONFIG_KEY") > "$POINTS_JSON"
MODEL="$("$LAUNCH_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))[0]["model"])' "$POINTS_JSON")"
MODEL_DIR="${MODEL_HOST_DIR%/}"
[[ -f "$MODEL_DIR/model.safetensors.index.json" ]] || fail "staged checkpoint not found: $MODEL_DIR"

# ---- Private runner config (cluster id stays mi355x-amds for launch policy) ----
RUNNER_CONFIG="$RUN_DIR/runners.yaml"
EXCLUDE_LINE=""
if [[ -n "$SLURM_EXCLUDE_NODES" ]]; then
    EXCLUDE_LINE="      exclude: [${SLURM_EXCLUDE_NODES}]"
fi
cat > "$RUNNER_CONFIG" <<EOF
labels:
  cluster:mi355x-amds:
    - ${RUNNER_NAME}
clusters:
  mi355x-amds:
    gpus-per-node: 8
    available-cpu-dram-mib: 3_095_781
    arch: x86_64
    env:
      MORI_RDMA_TC: "104"
    models:
      entries:
        ${MODEL##*/}: {root: model-weights, dir: ${MODEL_DIR##*/}}
    scheduler: slurm
    slurm:
      partition: compute
      exclusive: true
      gres: "gpu:{gpus}"
      cpus-per-task: 128
      srun-args: [--container-remap-root, --container-writable]
${EXCLUDE_LINE}
      volumes:
        hf-hub-cache: {path: ${HF_HUB_CACHE_HOST}}
        shared-hf-hub-cache: {path: /it-share/hf-hub-cache}
        aiperf-cache: {path: /it-share/aiperf-cache}
        it-share-data: {path: /it-share/data}
        model-weights: {path: ${MODEL_DIR%/*}}
      squash:
        dir: ${SQUASH_DIR}
        visibility: shared
        import: compute
        multi-node-import: false
      srt-slurm:
        network-interface: eno0
        single-node-time-limit: 500
        single-node-models: staged
        gpus-per-node-directive: true
        segment-directive: false
        outputs: ${SRT_OUTPUTS}
        host-setup:
          script: runners/srt-slurm/hooks/local-smoke/setup.sh
          timeout-s: 1200
          nodes: all
        volume-mounts:
          shared-hf-hub-cache: /hf_hub_cache/hub
        mounts:
          /dev/kfd: /dev/kfd
          /dev/dri: /dev/dri
          /it-share/hf_home: /it-share/hf_home
        extra:
          visible_devices_env: ROCR_VISIBLE_DEVICES
          default_gpu_exporter: null
          nginx_raise_ulimit: false
EOF

echo "======================================================"
echo " Qwen3.8-Flash-Next MXFP4 SGLang MTP AgentX (srt-slurm) — MI355X"
echo "======================================================"
echo "  Config:        $CONFIG_FILE : $CONFIG_KEY"
echo "  Image:         $IMAGE"
echo "  Squash:        $SQUASH_FILE"
echo "  Model:         $MODEL (staged at $MODEL_DIR)"
echo "  CONC_LIST:     $CONC_LIST"
echo "  Duration:      ${DURATION:-config default}"
echo "  THINKING_MODE: $THINKING_MODE"
echo "  Flags:         ECF=$EXACT_CHUNK_FILL GDN_CONV=$AITER_GDN_CONV GDN_DECODE=$AITER_GDN_DECODE HC_MIX=$AITER_HC_MIX QSA_PA_DECODE=$AITER_QSA_PA_DECODE GDR_FLYDSL=$AITER_GDR_FLYDSL TOPK_GATING=$AITER_TOPK_GATING GDN_PREFILL=$AITER_GDN_PREFILL GDN_PREFILL_H=$AITER_GDN_PREFILL_H HONOR_MEM_FRACTION=$AITER_HONOR_EXPLICIT_MEM_FRACTION"
echo "  Runner:        $RUNNER_NAME (private config $RUNNER_CONFIG)"
echo "  Run dir:       $RUN_DIR"
echo "  DRY_RUN:       $DRY_RUN"
echo "======================================================"

# Writes the point's CI environment (shell exports) for CONC, or fails if the
# config's conc-list does not contain it.
point_env() {
    "$LAUNCH_PYTHON" - "$POINTS_JSON" "$1" "$DURATION" <<'PY'
import json, shlex, sys
points, conc, duration = json.load(open(sys.argv[1])), int(sys.argv[2]), sys.argv[3]
match = [p for p in points if p["conc"] == conc]
if len(match) != 1:
    sys.exit(f"conc {conc} is not a single point of the config (have {[p['conc'] for p in points]})")
p = match[0]
env = {
    "EXP_NAME": p["exp-name"], "MODEL": p["model"], "MODEL_PREFIX": p["model-prefix"],
    "PRECISION": p["precision"], "FRAMEWORK": p["framework"], "RUNNER_TYPE": p["runner"],
    "TP": p["tp"], "PP_SIZE": p["pp"], "DCP_SIZE": p["dcp-size"], "PCP_SIZE": p["pcp-size"],
    "EP_SIZE": p["ep"], "DP_ATTENTION": str(p["dp-attn"]).lower(),
    "SPEC_DECODING": p["spec-decoding"], "CONC": p["conc"], "SRT_RECIPE": p["srt-recipe"],
    "KV_OFFLOADING": p["kv-offloading"],
    "KV_OFFLOAD_BACKEND": (p.get("kv-offload-backend") or {}).get("name", ""),
    "TOTAL_CPU_DRAM_GB": p["total-cpu-dram-gb"], "DURATION": duration or p["duration"],
    "CONFIG_IMAGE": p["image"],
}
for key, value in env.items():
    print(f"export {key}={shlex.quote(str(value))}")
PY
}

# Adds the feature flags to the server env of the workspace's recipe copy.
patch_recipe() {
    "$LAUNCH_PYTHON" - "$1" <<PY
import sys, yaml
path = sys.argv[1]
recipe = yaml.safe_load(open(path))
recipe["base"]["roles"]["agg"].setdefault("env", {}).update({
    "SGLANG_EXACT_CHUNK_FILL": "${EXACT_CHUNK_FILL}",
    "SGLANG_AITER_GDN_CONV": "${AITER_GDN_CONV}",
    "SGLANG_AITER_GDN_DECODE": "${AITER_GDN_DECODE}",
    "SGLANG_AITER_HC_MIX": "${AITER_HC_MIX}",
    "SGLANG_AITER_QSA_PA_DECODE": "${AITER_QSA_PA_DECODE}",
    "AITER_GDR_FLYDSL": "${AITER_GDR_FLYDSL}",
    "SGLANG_ROCM_USE_AITER_TOPK_GATING": "${AITER_TOPK_GATING}",
    "SGLANG_AITER_GDN_PREFILL": "${AITER_GDN_PREFILL}",
    "SGLANG_AITER_GDN_PREFILL_H": "${AITER_GDN_PREFILL_H}",
    "SGLANG_AITER_HONOR_EXPLICIT_MEM_FRACTION": "${AITER_HONOR_EXPLICIT_MEM_FRACTION}",
})
yaml.safe_dump(recipe, open(path, "w"), sort_keys=False, width=200)
PY
}

# ---- Run each concurrency on a fresh server ----
_rc=0
for C in $CONC_LIST; do
    echo ""
    echo "==================== CONC=$C ===================="
    POINT_DIR="$RUN_DIR/conc${C}"
    WS="$POINT_DIR/workspace"
    mkdir -p "$POINT_DIR"

    rsync -a --exclude '__pycache__' --exclude '/srt-single.*' --exclude '/results' \
        --exclude '/*.json' --exclude '/*.tar.gz' "$E2E_DIR/" "$WS/"
    # The submodule's .git is a relative gitdir pointer; anchor it to the repo.
    echo "gitdir: $(git -C "$E2E_DIR/utils/srt-slurm" rev-parse --absolute-git-dir)" > "$WS/utils/srt-slurm/.git"
    git -C "$INFERENCEX_DIR" rev-parse HEAD > "$POINT_DIR/inferencex-commit.txt"
    # The CI hook's GPU drain gate without its sudo hugepage steps: VRAM leaked
    # outside any process makes the PLE fp8 swap OOM at model load.
    mkdir -p "$WS/runners/srt-slurm/hooks/local-smoke"
    cat > "$WS/runners/srt-slurm/hooks/local-smoke/setup.sh" <<'EOF'
#!/usr/bin/env bash
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
echo "GPU memory on $(hostname):"
rocm-smi --showmemuse 2>/dev/null | grep -E "VRAM%" || true
wait_for_amd_gpu_clean
EOF

    # errexit is ignored in a subshell tested with ||, so take its status here.
    set +e
    (
        set -euo pipefail
        point_exports="$(point_env "$C")"
        eval "$point_exports"
        patch_recipe "$WS/$SRT_RECIPE"
        cp "$WS/$SRT_RECIPE" "$POINT_DIR/recipe.yaml"
        echo "  config image $CONFIG_IMAGE -> $IMAGE"

        cd "$WS"
        # A set CONC_LIST makes the client name its result ${RESULT_FILENAME}_conc<N>
        # (the multi-node layout), which the single-node collector does not look for.
        unset CONC_LIST
        export PYTHONPATH="$WS" PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/inferencex-pycache
        export GITHUB_WORKSPACE="$WS" RUNNER_NAME IMAGE THINKING_MODE
        export IS_AGENTIC=1 IS_MULTINODE=false DISAGG=false SCENARIO_TYPE=agentic-coding
        export ISL=0 OSL=0 MAX_MODEL_LEN=0 RANDOM_RANGE_RATIO=0.8
        export RUN_EVAL=false EVAL_ONLY=false
        export PORT=8888 INFMAX_CONTAINER_WORKSPACE=/workspace RESULT_DIR=/workspace/results
        export MODEL_DOWNLOAD_LOCK_TIMEOUT=21600 ENROOT_IMPORT_TIME_LIMIT=120 SALLOC_TIME_LIMIT=480
        export KEEP_LOGS=0 HF_HUB_CACHE=/mnt/hf_hub_cache/
        export AIPERF_EXPERIMENTAL_FAST=0 AIPERF_FAILED_REQUEST_THRESHOLD=0.10
        export INFERENCEX_RESULTS_PYTHON="$LAUNCH_PYTHON" GITHUB_STEP_SUMMARY=''
        export RESULT_FILENAME_BASE="${EXP_NAME}_${PRECISION}_${FRAMEWORK}_tp${TP}-pp${PP_SIZE}-dcp${DCP_SIZE}-pcp${PCP_SIZE}-ep${EP_SIZE}-dpa${DP_ATTENTION}_disagg-${DISAGG}_spec-${SPEC_DECODING}_conc${CONC}_${RUNNER_NAME}"
        RESULT_FILENAME="$("$LAUNCH_PYTHON" -P -m infx.results.result_filename)"
        export RESULT_FILENAME
        export GPU_COUNT=$((TP * PP_SIZE * PCP_SIZE))
        echo "$RESULT_FILENAME" > "$POINT_DIR/result_filename.txt"
        source benchmarks/runtime_settings.sh

        # Binds the patched recipe to this point exactly as srt-slurm will.
        "$LAUNCH_PYTHON" -P - <<'PY'
import os
from infx.srt_slurm.single_node import select_recipe
name, _ = select_recipe(os.environ["SRT_RECIPE"], os.environ)
print(f"  bound recipe variant: {name}")
PY
        echo "  RESULT_FILENAME: $RESULT_FILENAME"
        if [[ "$DRY_RUN" == 1 ]]; then
            echo "  DRY_RUN=1: not submitting"
            exit 0
        fi

        "$LAUNCH_PYTHON" -P -m infx.launch --runner-config "$RUNNER_CONFIG" cleanup
        set +e
        "$LAUNCH_PYTHON" -P -m infx.launch --runner-config "$RUNNER_CONFIG" run 2>&1 | tee "$POINT_DIR/launch.log"
        launch_rc=${PIPESTATUS[0]}
        set -e
        echo "  infx.launch run rc=$launch_rc"

        [[ -f "$RESULT_FILENAME.json" ]] && cp "$RESULT_FILENAME.json" "$POINT_DIR/"
        if [[ -d results/aiperf_artifacts ]]; then
            "$LAUNCH_PYTHON" -P -m infx.results.agentic.validate_agentic_result \
                results/aiperf_artifacts --failed-request-threshold "$AIPERF_FAILED_REQUEST_THRESHOLD" \
                || launch_rc=1
        fi
        exit "$launch_rc"
    )
    point_rc=$?
    set -e
    if [[ $point_rc -ne 0 ]]; then
        echo "  CONC=$C failed (rc=$point_rc)" >&2
        _rc=1
    fi
done

[[ "$DRY_RUN" == 1 ]] && { echo "Dry run done: $RUN_DIR"; exit "$_rc"; }

# ---- Summary ----
echo ""
echo "================ Qwen3.8-Flash-Next MXFP4 MTP AgentX Results ================"
"$LAUNCH_PYTHON" - "$RUN_DIR" <<'PY'
import glob, json, os, sys

def find(node, key):
    if isinstance(node, dict):
        if key in node:
            return node[key]
        for value in node.values():
            if (found := find(value, key)) is not None:
                return found
    return None

rows = []
for stem_file in sorted(glob.glob(os.path.join(sys.argv[1], "conc*", "result_filename.txt"))):
    point = os.path.dirname(stem_file)
    result = os.path.join(point, open(stem_file).read().strip() + ".json")
    if not os.path.exists(result):
        print(f"  {os.path.basename(point)}: no result JSON")
        continue
    d = json.load(open(result))
    rm = d.get("request_metrics", {})
    intvty = find(rm, "intvty") or {}
    per_gpu = rm.get("throughput", {}).get("per_gpu", {})
    rows.append((int(os.path.basename(point)[4:]), intvty.get("p50"), intvty.get("p90"),
                 per_gpu.get("total_tput_tps"), per_gpu.get("output_tput_tps")))

fmt = lambda v: f"{v:.2f}" if isinstance(v, (int, float)) else "n/a"
print(f"  {'conc':>5} | {'p50 intvty':>10} | {'p90 intvty':>10} | {'total/GPU tok/s':>15} | {'out/GPU tok/s':>13}")
for conc, p50, p90, tot, out in sorted(rows):
    print(f"  {conc:>5} | {fmt(p50):>10} | {fmt(p90):>10} | {fmt(tot):>15} | {fmt(out):>13}")
print("  p90 interactivity is the slow tail (1 / p90 ITL), so it is below p50.")
PY

echo ""
echo "Done. Results in: $RUN_DIR"
exit "$_rc"
