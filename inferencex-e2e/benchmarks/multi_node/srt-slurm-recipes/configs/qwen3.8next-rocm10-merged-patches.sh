#!/usr/bin/env bash
# Qwen3.8-Flash-Next baseline on lmsysorg/sglang-rocm:v0.5.22-rocm10-mi35x-20261018:
# only merged upstream PRs plus the monokernel, applied to the image's editable
# AITER (2bb54910) and SGLang (8dee3f6b) checkouts.
#   - AITER, each the PR's merge diff, in merge order:
#       ROCm/aiter#6201: gfx950 BF16 GEMM configs for Qwen3.8-Flash-Next
#       ROCm/aiter#5804: FlyDSL EP MoE (kmk3) and EP MoE GEMM2
#       ROCm/aiter#6467: reverse route indices initialized before Opus sorting
#       ROCm/aiter#6165: fused decode MoE route-sort + MXFP4 activation quant
#         (instantiated for hidden 4096 only, so Qwen3.8's 2560 keeps the
#         separate sort and quant)
#       ROCm/aiter#6225: a4w4 aux-sort fused/Opus crossover at 1.33 rows per expert
#       ROCm/aiter#6050: HY4 MXFP8 heterogeneous MoE fusion, without its .github/
#         CI files
#       ROCm/aiter#5993: FlyDSL hc_mix kernel (its SGLang dispatch, #41880, is not
#         applied, so nothing calls it)
#   - SGLang: sgl-project/sglang#45353, the opt-in FlyDSL monokernel for MTP target
#     verify and draft of up to 8 requests (SGLANG_ENABLE_QWEN4_MONOKERNEL,
#     _MTP_MONOKERNEL); verify needs the per-draft intermediate SSM states, which
#     --enable-linear-replayssm-spec drops
# The image's SGLang already has the merged Qwen3.8 PRs #41282 and #43383.
# All Python except #6165 and #6467, whose MoE sorting module is removed so AITER's
# JIT rebuilds it on first use. A patch that no longer applies fails the worker.
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

apply /sgl-workspace/aiter aiter-6201-qwen38-bf16-gemm-configs.patch
apply /sgl-workspace/aiter aiter-5804-ep-moe-gemm2.patch
apply /sgl-workspace/aiter aiter-6467-opus-sort-reverse-route-init.patch
apply /sgl-workspace/aiter aiter-6165-moe-sort-mxfp4-quant.patch
apply /sgl-workspace/aiter aiter-6225-aux-sort-crossover.patch
apply /sgl-workspace/aiter aiter-6050-hy4-mxfp8-moe-fusion.patch
apply /sgl-workspace/aiter aiter-5993-flydsl-hc-mix.patch
rm -rf /sgl-workspace/aiter/aiter/jit/module_moe_sorting_opus.so \
    /sgl-workspace/aiter/aiter/jit/build/module_moe_sorting_opus
apply /sgl-workspace/sglang sglang-qwen38-monokernel.patch

echo "$digest" > "$marker"
