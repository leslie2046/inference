#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <version> [docker build options...]" >&2
  exit 2
fi

version=$1
shift

image_repository=${IMAGE_REPOSITORY:-leslie2046/xinference}
image="${image_repository}:${version}-cu124"
proxy_url=${PROXY_URL:-http://192.168.0.188:21026}

build_args=(
  --progress=plain
  --memory=8g
  --build-arg BUILDKIT_INLINE_CACHE=1
  --cache-from "$image"
  --tag "$image"
  --file xinference/deploy/docker/Dockerfile.cu124
)

if [[ -n "$proxy_url" ]]; then
  build_args+=(
    --build-arg "http_proxy=$proxy_url"
    --build-arg "https_proxy=$proxy_url"
  )
fi

if [[ -n "${PIP_INDEX:-}" ]]; then
  build_args+=(--build-arg "PIP_INDEX=$PIP_INDEX")
fi

if [[ -n "${TORCH_INDEX:-}" ]]; then
  build_args+=(--build-arg "TORCH_INDEX=$TORCH_INDEX")
fi

DOCKER_BUILDKIT=1 docker build "${build_args[@]}" "$@" .
