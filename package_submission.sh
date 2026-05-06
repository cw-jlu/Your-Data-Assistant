#!/bin/bash
set -e

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <team_id> <version>"
    echo "Example: $0 team0042 1"
    exit 1
fi

TEAM_ID=$1
VERSION=$2
IMAGE_NAME="${TEAM_ID}:v${VERSION}"
ARCHIVE_NAME="${TEAM_ID}_v${VERSION}.tar.gz"

echo "Building Docker image: ${IMAGE_NAME} for linux/amd64..."
docker build --platform linux/amd64 -t "${IMAGE_NAME}" .

echo "Saving image to archive: ${ARCHIVE_NAME}..."
docker save "${IMAGE_NAME}" | gzip > "${ARCHIVE_NAME}"

echo "Done! The submission archive is ready: ${ARCHIVE_NAME}"
echo "File size:"
du -h "${ARCHIVE_NAME}"
