#!/usr/bin/env bash
# Build a Docker image and emit a Phase-1 submission tarball.
#
# Usage:
#   bash scripts/build_submission.sh <version>
#   e.g. bash scripts/build_submission.sh v1
#
# Output:
#   submissions/dabench_<version>.tar.gz  (must be ≤ 10 GB by rules)
#
# Side effects: prints image size, gzip size, and the MD5/SHA256 hashes
# you can include in the submission email.

set -euo pipefail

if [[ "${1:-}" == "" ]]; then
    echo "usage: bash scripts/build_submission.sh <version>" >&2
    echo "  example: bash scripts/build_submission.sh v1" >&2
    exit 1
fi
VERSION="$1"
IMAGE="dabench:${VERSION}"
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

OUT_DIR="${ROOT_DIR}/submissions"
TARBALL="${OUT_DIR}/dabench_${VERSION}.tar.gz"
mkdir -p "${OUT_DIR}"

# Force linux/amd64 so cross-builds on Apple Silicon dev hosts land in
# the same architecture as the eval runtime (rules-compute = x86-64).
# `--load` pulls the buildx result into the local docker daemon so the
# subsequent `docker save` step can find the image by tag.
docker buildx inspect --bootstrap >/dev/null 2>&1 || true

echo ">> building ${IMAGE} (linux/amd64) ..."
docker buildx build --platform linux/amd64 --tag "${IMAGE}" --load .

# Hard arch sanity check — never let a stray ARM image escape the build
# script. Even with the Dockerfile platform pin, a misconfigured buildx
# builder or a manual `docker build --platform` override could in theory
# slip an aarch64 manifest through; this gate makes that impossible.
IMAGE_ARCH="$(docker image inspect "${IMAGE}" --format='{{.Architecture}}')"
IMAGE_OS="$(docker image inspect "${IMAGE}" --format='{{.Os}}')"
if [[ "${IMAGE_OS}/${IMAGE_ARCH}" != "linux/amd64" ]]; then
    echo "ERROR: built image is ${IMAGE_OS}/${IMAGE_ARCH}, expected linux/amd64." >&2
    echo "       The eval runtime is x86-64 only (rules §compute)." >&2
    echo "       Check Dockerfile FROM --platform pin and your buildx builder." >&2
    exit 4
fi
echo ">> arch check: ${IMAGE_OS}/${IMAGE_ARCH} ✓"

IMAGE_SIZE_BYTES="$(docker image inspect "${IMAGE}" --format='{{.Size}}')"
IMAGE_SIZE_GB="$(awk -v b="${IMAGE_SIZE_BYTES}" 'BEGIN{printf "%.2f", b/1024/1024/1024}')"
echo ">> uncompressed image size: ${IMAGE_SIZE_GB} GB"

if [[ -f "${TARBALL}" ]]; then
    echo ">> removing existing ${TARBALL}"
    rm "${TARBALL}"
fi

echo ">> saving image to ${TARBALL} ..."
docker save "${IMAGE}" | gzip --best > "${TARBALL}"

TAR_SIZE_BYTES="$(stat -f%z "${TARBALL}" 2>/dev/null || stat -c%s "${TARBALL}")"
TAR_SIZE_GB="$(awk -v b="${TAR_SIZE_BYTES}" 'BEGIN{printf "%.2f", b/1024/1024/1024}')"
echo ">> tarball size: ${TAR_SIZE_GB} GB"

LIMIT_BYTES=$((10 * 1024 * 1024 * 1024))
if (( TAR_SIZE_BYTES > LIMIT_BYTES )); then
    echo "ERROR: tarball exceeds 10 GB limit (${TAR_SIZE_GB} GB). Submission will be rejected." >&2
    exit 2
fi

if command -v shasum >/dev/null 2>&1; then
    SHA256="$(shasum -a 256 "${TARBALL}" | awk '{print $1}')"
elif command -v sha256sum >/dev/null 2>&1; then
    SHA256="$(sha256sum "${TARBALL}" | awk '{print $1}')"
else
    SHA256="(install shasum or sha256sum to record hash)"
fi
echo ">> sha256: ${SHA256}"

cat <<EOF

submission ready:
  file:      ${TARBALL}
  size:      ${TAR_SIZE_GB} GB (image ${IMAGE_SIZE_GB} GB)
  sha256:    ${SHA256}

next steps:
  1. upload to Google Drive with "Anyone with the link can view"
  2. email the share link + sha256 to the organizers
  3. record the submission in docs/SUBMISSION_LOG.md
EOF
