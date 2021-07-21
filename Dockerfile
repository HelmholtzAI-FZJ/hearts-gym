# syntax=docker/dockerfile:1
ARG BASE_IMG=ubuntu
ARG VERSION=latest
FROM $BASE_IMG:$VERSION
ARG REPO_DIR=/opt/hearts-gym
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
RUN apt-get update && apt-get install -y \
	python3 \
	python3-pip \
	python-is-python3 \
	git \
	zlib1g \
	libglib2.0-0 \
	emacs joe mg nano neovim vim vis zile
COPY . $REPO_DIR
WORKDIR $REPO_DIR
RUN python -m pip install --upgrade pip
RUN python -m pip install --upgrade tensorflow==2.5.0
RUN python -m pip install \
	-f https://download.pytorch.org/whl/torch_stable.html \
	torch==1.9.0+cpu \
	torchvision==0.10.0+cpu
RUN python -m pip install --upgrade jax jaxlib
RUN python -m pip install -e .
