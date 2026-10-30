#!/usr/bin/env bash
# Qwen3.8-Flash-Next baseline on lmsysorg/sglang-rocm:v0.5.23-rocm10-mi35x-20261030:
# merged PRs that the image does not have yet (hc_mix and the GEMM tables in AITER,
# the monokernel in sgl-amd-mono), applied to the image's editable AITER (285579b6);
# the sgl-amd-mono checkout is moved to the merge commit. No open PR is patched in, and SGLang (17e94c4677) is
# not patched: it has the monokernel interface and KV layer router (sglang#46980),
# the spec registration by model type (sglang#46036) and hc_mix (sglang#41880).
#   - AITER, merged after the image's commit:
#       ROCm/aiter#5993: FlyDSL hc_mix kernel (merge diff), dispatched by
#         sglang#41880
#       ROCm/aiter#6769: gfx950 A8W8 bpreshuffle and BF16 GEMM tunings for
#         Qwen3.8-Flash-Next (model_configs/, merged in by default), as the diff
#         of the PR head that was merged
#   - sgl-amd-mono: sgl-project/sgl-amd-mono#29, merged as b10f3b71: the FlyDSL
#     monokernel for MTP target verify and draft of up to 8 requests
#     (--enable-amd-monokernel). The image pins sgl-amd-mono at 90c23551, so the
#     editable checkout is moved to b10f3b71 (fetch from GitHub; skipped when the
#     checkout already has qwen38_flash_next/spec.py). Verify needs the per-draft
#     intermediate SSM states, which --enable-linear-replayssm-spec drops, and the
#     PLE table on the GPU (no --ple-offload-embedding).
# The image's AITER already has ROCm/aiter#6201, #5804, #6467, #6165, #6225,
# #6050, #5992 and #6670, and its SGLang the merged Qwen3.8 PRs #41282 and #43383.
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

# sgl-amd-mono is an editable git install (pip -e git+...), so the package's
# checkout is wherever the image cloned it; find_spec does not import it.
mono="$(python3 - <<'PY'
import importlib.util
import os
spec = importlib.util.find_spec("sgl_amd_mono")
locations = list(spec.submodule_search_locations or []) if spec else []
print(os.path.dirname(os.path.realpath(locations[0])) if locations else "")
PY
)"
if [[ -z $mono ]] || [[ "$(git -c safe.directory="*" -C "$mono" rev-parse --show-toplevel 2>/dev/null)" != "$mono" ]]; then
    echo "qwen3.8next: sgl_amd_mono is not an editable git checkout (found: '${mono}')" >&2
    exit 1
fi
echo "qwen3.8next: sgl-amd-mono checkout $mono at $(git -c safe.directory="*" -C "$mono" rev-parse --short=8 HEAD)"

apply /sgl-workspace/aiter aiter-5993-flydsl-hc-mix.patch
apply /sgl-workspace/aiter aiter-6769-qwen38-gemm-tunings.patch
# The pin is the merge commit of sgl-amd-mono#29. The checkout is an editable
# install, so moving it is the reinstall (pure Python, no dependency change).
mono_commit=b10f3b71c45a70f504c5cad2d622c3e423196e90
if [[ -f $mono/sgl_amd_mono/qwen38_flash_next/spec.py ]]; then
    echo "qwen3.8next: sgl-amd-mono already has qwen38_flash_next; not moving the checkout"
else
    for attempt in 1 2 3; do
        git -c safe.directory="*" -C "$mono" fetch --quiet origin "$mono_commit" && break
        [[ $attempt -lt 3 ]] || { echo "qwen3.8next: cannot fetch sgl-amd-mono $mono_commit" >&2; exit 1; }
        sleep 5
    done
    git -c safe.directory="*" -C "$mono" checkout --quiet --detach "$mono_commit"
    echo "qwen3.8next: sgl-amd-mono checked out at $(git -c safe.directory="*" -C "$mono" rev-parse --short=8 HEAD)"
fi

echo "$digest" > "$marker"
