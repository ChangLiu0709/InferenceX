#!/usr/bin/env bash
# Qwen3.8-Flash-Next baseline on lmsysorg/sglang-rocm:v0.5.22-rocm10-mi35x-20261026:
# the monokernel and the hc_mix, GEMM table, QSA and GDN decode PRs, applied to the
# image's editable AITER (285579b6) and SGLang (5ef99013) checkouts.
#   - AITER, merged after the image's commit:
#       ROCm/aiter#5993: FlyDSL hc_mix kernel (merge diff), dispatched by
#         sglang#41880
#       ROCm/aiter#5992: FlyDSL causal conv1d prefill and GDN decode/verify
#       ROCm/aiter#6670: fused FlyDSL QSA sparse decode
#       ROCm/aiter#6769: gfx950 A8W8 bpreshuffle and BF16 GEMM tunings for
#         Qwen3.8-Flash-Next (model_configs/, merged in by default)
#     (#5992, #6670 and #6769 as the diff of the PR head that was merged)
#   - SGLang: sgl-project/sglang#45353, the opt-in FlyDSL monokernel for MTP target
#     verify and draft of up to 8 requests (SGLANG_ENABLE_QWEN4_MONOKERNEL,
#     _MTP_MONOKERNEL); verify needs the per-draft intermediate SSM states, which
#     --enable-linear-replayssm-spec drops
#   - SGLang, open PRs:
#       sgl-project/sglang#41879: opt-in AITER FlyDSL prefill conv and GDN decode
#         (SGLANG_AITER_GDN_CONV, SGLANG_AITER_GDN_DECODE), on the image's
#         gdn_backend.py imports
#       sgl-project/sglang#41880: FlyDSL hc_mix on gfx950 (on whenever AITER has it)
#         and the single-kernel hc_combine on ROCm
#       sgl-project/sglang#45196: QSA decode through the fused FlyDSL sparse decode
#         (SGLANG_ENABLE_QSA_FLYDSL_DECODE, default on)
# The image's AITER already has ROCm/aiter#6201, #5804, #6467, #6165, #6225 and
# #6050, and its SGLang the merged Qwen3.8 PRs #41282 and #43383.
# All Python. A patch that no longer applies fails the worker.
set -euo pipefail
# srtctl only looks up setup_script here; the patches live with the single-node
# recipe and are read through the workspace mount.
patches=/infmax-workspace/benchmarks/single_node/srt-slurm-recipes/qwen3.8next/sglang/mi355x-fp4-mtp/patches
[[ -d $patches ]] || { echo "qwen3.8next: $patches not found" >&2; exit 1; }
# Later patches touch files of earlier ones, so a per-patch reverse check cannot
# tell a re-run apart; the marker records the patch set a container already has.
marker=/sgl-workspace/.qwen3.8next-rocm10-merged-patches
digest="$(cat "${BASH_SOURCE[0]}" "$patches"/*.patch | sha256sum | cut -d' ' -f1)"
if [[ -f $marker ]]; then
    if [[ "$(cat "$marker")" == "$digest" ]]; then
        echo "qwen3.8next merged patches already applied"
        exit 0
    fi
    echo "qwen3.8next: $marker is from a different patch set" >&2
    exit 1
fi
if [[ -f /sgl-workspace/.qwen3.8next-rocm10-patches ]]; then
    echo "qwen3.8next: container already has the full patch set" >&2
    exit 1
fi

apply() {
    git -c safe.directory="*" -C "$1" apply "$patches/$2"
    echo "qwen3.8next patch applied: $2"
}

apply /sgl-workspace/aiter aiter-5993-flydsl-hc-mix.patch
apply /sgl-workspace/aiter aiter-5992-flydsl-gdn-conv-decode.patch
apply /sgl-workspace/aiter aiter-6670-flydsl-qsa-sparse-decode.patch
apply /sgl-workspace/aiter aiter-6769-qwen38-gemm-tunings.patch
apply /sgl-workspace/sglang sglang-qwen38-monokernel.patch
apply /sgl-workspace/sglang sglang-41879-aiter-gdn-conv-decode-hooks.patch
apply /sgl-workspace/sglang sglang-41880-flydsl-hc-mix.patch
apply /sgl-workspace/sglang sglang-45196-qsa-flydsl-sparse-decode.patch

echo "$digest" > "$marker"
