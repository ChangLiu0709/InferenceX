#!/usr/bin/env bash
# Qwen3.8-Flash-Next kernels on lmsysorg/sglang-rocm:v0.5.21-rocm10-mi35x-20261014,
# applied to its editable AITER (e7d2453f) and SGLang (bab9c81f) checkouts:
#   - ROCm/aiter#5992: FlyDSL causal conv1d prefill and GDN decode/verify
#   - sgl-project/sglang#41879: opt-in SGLANG_AITER_GDN_CONV/_DECODE hooks
#   - fused Triton QK Gemma-RMSNorm + M-RoPE + gate on HIP
# Python only, so nothing is rebuilt. A patch that no longer applies fails the worker.
set -euo pipefail
patches="$(dirname "${BASH_SOURCE[0]}")/qwen3.8next-rocm10-patches"

apply() {
    local repo=$1 patch=$patches/$2
    if git -c safe.directory="*" -C "$repo" apply --reverse --check "$patch" 2>/dev/null; then
        echo "qwen3.8next patch already applied: $2"
    else
        git -c safe.directory="*" -C "$repo" apply "$patch"
        echo "qwen3.8next patch applied: $2"
    fi
}

apply /sgl-workspace/aiter aiter-flydsl-gdn-conv-decode.patch
apply /sgl-workspace/sglang sglang-aiter-gdn-conv-decode-hooks.patch
apply /sgl-workspace/sglang sglang-fused-qk-rope-gate-hip.patch
