# Base image.
FROM nvidia/cuda:11.2.0-runtime-ubuntu20.04

# Non-interactive mode.
ARG DEBIAN_FRONTEND=noninteractive

# Install dependencies.
RUN apt update; \
    apt upgrade -y; \
    apt install -y \
        curl \
        git \
        python3-opencv \
        s3cmd \
        software-properties-common \
        wget

RUN add-apt-repository ppa:deadsnakes/ppa; \
    apt update; \
    apt install -y python3.10

RUN curl -sSL https://install.python-poetry.org | python3.10 -
# RUN export PATH="/root/.local/bin:$PATH"

# Add the local code.
RUN git clone https://github.com/picklelo/xist.git /xist

# Install the dependencies. 
WORKDIR /xist
RUN /root/.local/bin/poetry install; \
    /root/.local/bin/poetry run pip install -e \
    git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers \
    git+https://github.com/openai/CLIP.git@main#egg=clip
RUN /root/.local/bin/poetry run pip uninstall -y torch torchvision; \
    /root/.local/bin/poetry run pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116
