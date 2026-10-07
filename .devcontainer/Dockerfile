FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG COPPELIASIM_RELEASE=V4_6_0_rev18
ARG COPPELIASIM_PKG=CoppeliaSim_Edu_${COPPELIASIM_RELEASE}_Ubuntu22_04
ARG COPPELIASIM_URL=https://downloads.coppeliarobotics.com/${COPPELIASIM_RELEASE}/${COPPELIASIM_PKG}.tar.xz

ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=1000

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates wget xz-utils git sudo less nano bash-completion \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libgl1-mesa-dri libglx-mesa0 libegl1 libopengl0 libglu1-mesa mesa-utils \
        libglib2.0-0 libdbus-1-3 libfontconfig1 libfreetype6 \
        libx11-6 libx11-xcb1 libxext6 libxrender1 libxi6 libsm6 libice6 \
        libxkbcommon0 libxkbcommon-x11-0 \
        libxcb1 libxcb-glx0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
        libxcb-randr0 libxcb-render0 libxcb-render-util0 libxcb-shape0 \
        libxcb-shm0 libxcb-sync1 libxcb-util1 libxcb-xfixes0 libxcb-xinerama0 \
        libxcb-xkb1 libxcb-cursor0 \
        libzmq5 libsodium23 \
        libavcodec58 libavformat58 libavutil56 libswscale5 \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv python3-tk \
        python3-numpy python3-scipy python3-matplotlib python3-pandas python3-sympy \
        python3-zmq python3-cbor2 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m -s /bin/bash ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME} \
    && mkdir -p /opt/coppeliasim \
    && chown ${USER_UID}:${USER_GID} /opt/coppeliasim

RUN printf '#!/bin/sh\nexec /opt/coppeliasim/coppeliaSim.sh "$@"\n' > /usr/local/bin/coppeliasim \
    && chmod +x /usr/local/bin/coppeliasim

USER ${USERNAME}
RUN wget -q --show-progress "${COPPELIASIM_URL}" -O /tmp/coppeliasim.tar.xz \
    && tar -xJf /tmp/coppeliasim.tar.xz -C /opt/coppeliasim --strip-components=1 \
    && rm /tmp/coppeliasim.tar.xz

ENV COPPELIASIM_ROOT=/opt/coppeliasim \
    QT_X11_NO_MITSHM=1 \
    MPLBACKEND=TkAgg

WORKDIR /workspace
CMD ["bash"]
