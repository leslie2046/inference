#!/bin/bash
#docker buildx build   --memory=8g --build-arg http_proxy=http://192.168.0.188:21026   --build-arg https_proxy=http://192.168.0.188:21026   -t leslie2046/xinference:$1-cu124   -f xinference/deploy/docker/Dockerfile.cu124 .

DOCKER_BUILDKIT=1 docker build \
	--progress=plain \
	--memory=8g \
	--build-arg http_proxy=http://192.168.0.188:21026 \
	--build-arg https_proxy=http://192.168.0.188:21026 -t leslie2046/xinference:$1-cu124 -f xinference/deploy/docker/Dockerfile.cu124 \
		      .
