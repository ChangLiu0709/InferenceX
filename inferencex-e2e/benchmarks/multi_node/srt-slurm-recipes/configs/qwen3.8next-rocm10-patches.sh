#!/usr/bin/env bash
# Qwen3.8-Flash-Next kernels on lmsysorg/sglang-rocm:v0.5.21-rocm10-mi35x-20261014,
# applied to its editable AITER (e7d2453f) and SGLang (bab9c81f) checkouts:
#   - ROCm/aiter#5992: FlyDSL causal conv1d prefill and GDN decode/verify
#   - sgl-project/sglang#41879: opt-in SGLANG_AITER_GDN_CONV/_DECODE hooks
#   - fused Triton QK Gemma-RMSNorm + M-RoPE + gate on HIP
#   - FlyDSL hc_mix for the hyper-connection mix (SGLANG_AITER_HC_MIX, default
#     on) and the single-kernel hc_combine on ROCm (SGLANG_HC_COMBINE_SPLIT=0)
#   - opt-in FlyDSL paged decode for QSA attention (SGLANG_AITER_QSA_PA_DECODE)
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
# Python only, so nothing is rebuilt. A patch that no longer applies fails the worker.
set -euo pipefail
patches="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/qwen3.8next-rocm10-patches"
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
apply /sgl-workspace/sglang sglang-aiter-gdn-conv-decode-hooks.patch
apply /sgl-workspace/sglang sglang-fused-qk-rope-gate-hip.patch
apply /sgl-workspace/sglang sglang-flydsl-hc-mix.patch
apply /sgl-workspace/sglang sglang-flydsl-qsa-pa-decode.patch
apply /sgl-workspace/sglang sglang-aiter-gdn-prefill.patch
apply /sgl-workspace/sglang sglang-aiter-topk-gating.patch
apply /sgl-workspace/sglang sglang-gdn-ratio3-packing.patch
apply /sgl-workspace/sglang sglang-qsa-padded-group-mtp-shared-expert.patch
apply /sgl-workspace/sglang sglang-prefill-chunk-qsa-alignment.patch

echo "$digest" > "$marker"
