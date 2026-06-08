#!/usr/bin/env bash
set -euo pipefail

# rent a gpu from prime intellect for the dpo run.
# pick an id from `prime availability list --gpu-type H100_80GB --gpu-count 1`.
# avoid spot pods -- they get preempted mid-run.
# image set is provider-specific: massedcompute only ships ubuntu_22_cuda_12
# (no prebuilt torch), which is fine since `uv sync` installs the locked torch.
# datacrunch/crusoe support cuda_12_x_pytorch_2_x -- override with IMAGE=.

POD_ID="${1:-}"
NAME="${NAME:-paperhound-dpo}"
DISK="${DISK:-200}"
IMAGE="${IMAGE:-ubuntu_22_cuda_12}"

if [ -z "$POD_ID" ]; then
  echo "usage: bash scripts/rent_prime_pod.sh <availability-id>" >&2
  echo "see: prime availability list --gpu-type H100_80GB --gpu-count 1" >&2
  exit 1
fi

prime pods create \
  --id "$POD_ID" \
  --name "$NAME" \
  --disk-size "$DISK" \
  --image "$IMAGE" \
  --yes

echo "pod requested. list: prime pods list   |   ssh: prime pods ssh <pod-id>"
