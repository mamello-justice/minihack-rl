FROM lscr.io/linuxserver/webtop:ubuntu-xfce

RUN apt update
RUN apt install -yq\
    apt-transport-https \
    bison \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    flex \
    git \
    libbz2-dev \
    ninja-build \
    python3 \
    python3-pip \
    software-properties-common \
    wget

RUN python3 -m pip install nle
RUN python3 -m pip install minihack
RUN python3 -m pip install matplotlib

WORKDIR /app
COPY . .