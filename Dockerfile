# ==================================================================
# module list
# ------------------------------------------------------------------
# python        3.6    (apt)
# pytorch       latest (pip)
# ==================================================================

FROM nvidia/cudagl:11.1.1-devel-ubuntu20.04
ENV LANG C.UTF-8
ENV TZS=Europe/London
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# ==================================================================
# User setting
# ------------------------------------------------------------------

ARG USER
ARG PASSWD
ARG USER_ID

RUN apt-get update && \
    apt-get install -y sudo

ENV HOME /home/$USER

RUN groupadd -g $USER_ID -o $USER && \
    useradd -m $USER -u $USER_ID -g $USER_ID && \
    gpasswd -a $USER sudo && \
    echo "$USER:$PASSWD" | chpasswd && \
    echo 'Defaults visiblepw' >> /etc/sudoers && \
    echo "$USER ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER $USER
WORKDIR $HOME
ENV HOME $HOME

RUN APT_INSTALL="sudo apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \

    sudo rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \

    sudo apt-get update && \

# ==================================================================
# tools
# ------------------------------------------------------------------

    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        build-essential \
        apt-utils \
        ca-certificates \
        wget \
        git \
        vim \
        libssl-dev \
        curl \
        unzip \
        unrar \
        zsh \
        neovim \
        tmux \
        && \

    $GIT_CLONE https://github.com/Kitware/CMake ~/cmake && \
    cd ~/cmake && \
    ./bootstrap && \
    sudo make -j"$(nproc)" install

# ==================================================================
# zsh
# ------------------------------------------------------------------
RUN APT_INSTALL="sudo apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \

    zsh && \
    sudo chsh -s $(which zsh) && \

# ==================================================================
# python
# ------------------------------------------------------------------

    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        software-properties-common \
        && \
    sudo add-apt-repository ppa:deadsnakes/ppa && \
    sudo apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        python3-pip \
        python3.6 \
        python3.6-dev \
	python3.6-distutils \
        python3-distutils-extra \
        ffmpeg \
        libsm6 \
        libxext6 \
        && \
    wget -O ~/get-pip.py \
        https://bootstrap.pypa.io/pip/3.6/get-pip.py && \
    sudo python3.6 ~/get-pip.py && \
    python3.6 ~/get-pip.py && \
    sudo ln -s /usr/bin/python3.6 /usr/local/bin/python && \
    pip3 install --upgrade pip

RUN APT_INSTALL="sudo apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    #PIP_INSTALL="pip install" && \
    GIT_CLONE="git clone --depth 10" && \
    $PIP_INSTALL \
        setuptools && \
    $PIP_INSTALL \
        numpy \
        scipy \
        cloudpickle \
        Cython \
        tqdm \
        h5py \
        enum34 \
        pyyaml \
        wandb  \
        sympy  \
        opencv_python \
        funcsigs && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        freeglut3-dev \
        ninja-build \
        htop \
        swig \
        openssh-server \
        libeigen3-dev \
        libgl1-mesa-dev \
        libgl1-mesa-glx \
        libosmesa6-dev \
        patchelf \
        libopenmpi-dev \
        libglew-dev \
        libyaml-cpp-dev

# ==================================================================
# pytorch
# ------------------------------------------------------------------
RUN PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    $PIP_INSTALL \
    torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# ==================================================================
# zsh
# ------------------------------------------------------------------

RUN APT_INSTALL="sudo apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    #PIP_INSTALL="pip install" && \
    GIT_CLONE="git clone --depth 10" && \
    sh -c "$(wget -O- https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" && \
    $GIT_CLONE https://github.com/junjungoal/dotfiles.git && \
    sh dotfiles/dotfilesLink.sh && \
    $GIT_CLONE https://github.com/zsh-users/zsh-autosuggestions ~/.zsh/zsh-autosuggestions && \
    $(which zsh) -c "source ~/.zsh/zsh-autosuggestions/zsh-autosuggestions.zsh" && \
    $(which zsh) -c "source ~/.zshrc" && \
    rm -rf ~/.vim && \
    $GIT_CLONE https://github.com/VundleVim/Vundle.vim.git ~/.vim/bundle/Vundle.vim && \
    mkdir .config && \
    cp -r dotfiles/config/* .config/

ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics
ENV LIBGL_ALWAYS_INDIRECT 0

RUN ["/bin/zsh", "-c", \
 "nvim +VundleInstall +qall"]

RUN ["/bin/zsh", "-c", \
    "cd ~/", \
    "mkdir .mujoco", \
    "cd ./mujoco", \
    "wget https://www.roboti.us/download/mujoco200_linux.zip", \
    "unzip mujoco200_linux.zip", \
    "mv mujoco_200_linux mujoco200"]
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin

RUN ["/bin/zsh", "-c", \
    "cd ~/", \
    "git clone git@github.com:ompl/ompl.git", \
    "cd ./ompl", \
    "cmake .", \
    "sudo make install"]
# ==================================================================
# config & cleanup
# ------------------------------------------------------------------

RUN mkdir -p ~/projects

RUN sudo ldconfig && \
    sudo apt-get clean && \
    sudo apt-get autoremove && \
    sudo rm -rf /var/lib/apt/lists/* /tmp/*

