#!/usr/bin/env bash
# Qwen3.8-Flash-Next baseline on lmsysorg/sglang-rocm:v0.5.23-rocm10-mi35x-20261030.
# Nothing is patched: SGLang (17e94c4677) and AITER (285579b6) are used as shipped.
# SGLang has the monokernel interface and KV layer router (sglang#46979, #46980,
# #46990) and the spec registration by model type (sglang#46036), but pins
# sgl-amd-mono at 90c23551, before the Qwen3.8 monokernel. This script moves the
# image's editable sgl-amd-mono checkout to b10f3b71, the merge commit of
# sgl-project/sgl-amd-mono#29: the FlyDSL monokernel for MTP target verify and
# draft of up to 8 requests (--enable-amd-monokernel). Verify needs the
# per-draft intermediate SSM states, which --enable-linear-replayssm-spec drops,
# and the PLE table on the GPU (no --ple-offload-embedding). The step is skipped
# when the checkout already has qwen38_flash_next/spec.py.
set -euo pipefail
marker=/sgl-workspace/.qwen3.8next-sgl-amd-mono-29
digest="$(sha256sum "${BASH_SOURCE[0]}" | cut -d' ' -f1)"
if [[ -f $marker ]]; then
    if [[ "$(cat "$marker")" == "$digest" ]]; then
        echo "qwen3.8next setup already done"
        exit 0
    fi
    echo "qwen3.8next: $marker is from a different setup script" >&2
    exit 1
fi
if [[ -f /sgl-workspace/.qwen3.8next-rocm10-patches ]]; then
    echo "qwen3.8next: container already has the full patch set" >&2
    exit 1
fi

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

# Moving the editable checkout is the reinstall (pure Python, no dependency change).
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
