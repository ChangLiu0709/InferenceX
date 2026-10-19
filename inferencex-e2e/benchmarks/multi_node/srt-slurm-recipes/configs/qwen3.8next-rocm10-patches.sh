#!/usr/bin/env bash
# Qwen3.8-Flash-Next kernels on lmsysorg/sglang-rocm:v0.5.21-rocm10-mi35x-20261014,
# applied to its editable AITER (e7d2453f) and SGLang (bab9c81f) checkouts:
#   - ROCm/aiter#5992: FlyDSL causal conv1d prefill and GDN decode/verify
#   - sgl-project/sglang#41879: opt-in SGLANG_AITER_GDN_CONV/_DECODE hooks
#   - fused Triton QK Gemma-RMSNorm + M-RoPE + gate on HIP
#   - FlyDSL hc_mix for the hyper-connection mix up to 128 rows (SGLANG_AITER_HC_MIX,
#     default on) and the single-kernel hc_combine on ROCm (SGLANG_HC_COMBINE_SPLIT=0)
#   - opt-in FlyDSL paged decode for QSA attention (SGLANG_AITER_QSA_PA_DECODE), on
#     AITER mainline's planned pa_decode
#   - fused FlyDSL QSA sparse decode that reads KV from the pool
#     (SGLANG_ENABLE_QSA_FLYDSL_DECODE, default on, tried before the paged decode)
#   - fused GDN decode projection + conv1d on HIP and off-by-default GDN prefill
#     opt-ins (SGLANG_GDN_FUSED_QKV_NORM, _WIDE_QKV_SPLIT, _PACKED_QWEN4,
#     _HIP_PREFILL_VIEWS)
#   - tuned BF16 / A8W8 bpreshuffle GEMM tables for Qwen3.8 on gfx950 (AITER loads
#     model_configs/ by default)
#   - ROCm/aiter#5392: sigmoid gate for the fused gated RMSNorm + FP8 per-token
#     quant, which SGLANG_OPT_QWEN4_FUSED_GDN_NORM_QUANT needs for Qwen3.8's
#     sigmoid output gate
#   - off-by-default decode optimizations: shared per-token FP8 attention input
#     quant, fused GDN gated norm + out_proj quant, shared-expert gate folded into
#     the router, BF16 lm_head through tgemm (SGLANG_OPT_QWEN4_SHARED_ATTN_QUANT,
#     _QWEN4_FUSED_GDN_NORM_QUANT, _QWEN_MOE_FUSED_SHARED_TOPK, _AITER_LM_HEAD_GEMM)
#   - opt-in AITER GDN chunked prefill (SGLANG_AITER_GDN_PREFILL; only
#     SGLANG_AITER_GDN_PREFILL_H=fused works on this AITER)
#   - AITER topk_gating for the 512-expert router (SGLANG_ROCM_USE_AITER_TOPK_GATING,
#     default on)
#   - sgl-project/sglang#44302: fused QKVZ/BA packing for head ratio 3
#   - faster FlyDSL GDN decode/verify (used with SGLANG_AITER_GDN_DECODE)
#   - exact chunk fill (default on for gfx95) fixes: unfinished prefill chunks end
#     on a QSA page, padded QSA group writes read an inert source, and off-page
#     prefixes use the absolute mamba checkpoint grid
#   - fused shared-expert loading for Qwen3.8, strict for the MTP draft
# All but the upstream PRs come from amd-nprotaso/{aiter,sglang} qwen_3_8_improvement.
# All Python except aiter#5392, whose prebuilt module is removed so AITER's JIT
# rebuilds it (~10 s) on first use. A patch that no longer applies fails the worker.
set -euo pipefail
# srtctl only looks up setup_script here; the patches live with the single-node
# recipe and are read through the workspace mount.
patches=/infmax-workspace/benchmarks/single_node/srt-slurm-recipes/qwen3.8next/sglang/mi355x-fp4-mtp/patches
[[ -d $patches ]] || { echo "qwen3.8next: $patches not found" >&2; exit 1; }
# Later patches touch files of earlier ones, so a per-patch reverse check cannot
# tell a re-run apart; the marker records the patch set a container already has.
marker=/sgl-workspace/.qwen3.8next-rocm10-patches
digest="$(cat "${BASH_SOURCE[0]}" "$patches"/*.patch | sha256sum | cut -d' ' -f1)"
if [[ -f $marker ]]; then
    if [[ "$(cat "$marker")" == "$digest" ]]; then
        echo "qwen3.8next patches already applied"
        exit 0
    fi
    echo "qwen3.8next: $marker is from a different patch set" >&2
    exit 1
fi

apply() {
    git -c safe.directory="*" -C "$1" apply "$patches/$2"
    echo "qwen3.8next patch applied: $2"
}

apply /sgl-workspace/aiter aiter-flydsl-gdn-conv-decode.patch
apply /sgl-workspace/aiter aiter-flydsl-hc-mix.patch
apply /sgl-workspace/aiter aiter-flydsl-qsa-pa-decode.patch
apply /sgl-workspace/aiter aiter-flydsl-gdn-fused-prefill.patch
apply /sgl-workspace/aiter aiter-flydsl-gdn-decode-verify-perf.patch
apply /sgl-workspace/aiter aiter-qwen38-tuned-gemm-tables.patch
apply /sgl-workspace/aiter aiter-gated-rmsnorm-sigmoid.patch
rm -rf /sgl-workspace/aiter/aiter/jit/module_gated_rmsnorm_quant.so \
    /sgl-workspace/aiter/aiter/jit/build/module_gated_rmsnorm_quant
apply /sgl-workspace/sglang sglang-aiter-gdn-conv-decode-hooks.patch
apply /sgl-workspace/sglang sglang-fused-qk-rope-gate-hip.patch
apply /sgl-workspace/sglang sglang-flydsl-hc-mix.patch
apply /sgl-workspace/sglang sglang-flydsl-qsa-pa-decode.patch
apply /sgl-workspace/sglang sglang-aiter-gdn-prefill.patch
apply /sgl-workspace/sglang sglang-aiter-topk-gating.patch
apply /sgl-workspace/sglang sglang-gdn-ratio3-packing.patch
apply /sgl-workspace/sglang sglang-qsa-padded-group-mtp-shared-expert.patch
apply /sgl-workspace/sglang sglang-prefill-chunk-qsa-alignment.patch
apply /sgl-workspace/sglang sglang-qsa-flydsl-sparse-decode.patch
apply /sgl-workspace/sglang sglang-gdn-hip-fused-proj-conv.patch
apply /sgl-workspace/sglang sglang-qsa-pa-decode-mainline.patch
apply /sgl-workspace/sglang sglang-flydsl-hc-mix-128-rows.patch
apply /sgl-workspace/sglang sglang-qwen38-decode-opts.patch

echo "$digest" > "$marker"
