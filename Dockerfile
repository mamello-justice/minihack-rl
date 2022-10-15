FROM ubuntu:18.04

# Python and most build deps
RUN apt-get update && apt-get install -y \
    build-essential \
    autoconf \
    cmake \
    libtool \
    pkg-config \
    python3-dev \
    python3-pip \
    python3-numpy \
    git \
    flex \
    bison \
    libbz2-dev \
    wget

WORKDIR /app

COPY . .

# Install NetHack Learning Environment (Pull Submodule and install)
RUN cd /app/packages/nle
RUN pip install nle

# Install minihack
# TODO: Use as subdirectory
RUN pip install minihack

CMD [ "python", "minihack/main.py" ]