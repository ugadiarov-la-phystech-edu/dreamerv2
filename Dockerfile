FROM tensorflow/tensorflow:2.4.2-gpu

RUN   apt-key del 7fa2af80
ADD   https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb .
RUN   dpkg -i cuda-keyring_1.0-1_all.deb
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

# System packages.
RUN apt-get update && apt-get install -y \
  ffmpeg \
  libgl1-mesa-dev \
#  python3-pip \
  unrar \
  wget \
  && apt-get clean

ARG UNAME
ARG GID
ARG UID
RUN groupadd -g $GID -o $UNAME
RUN useradd -m -u $UID -g $GID -o -s /bin/bash --create-home $UNAME
USER $UNAME
WORKDIR /home/$UNAME

# MuJoCo.
ENV MUJOCO_GL egl
RUN mkdir -p ~/.mujoco && \
  wget -nv https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip && \
  unzip mujoco.zip -d ~/.mujoco && \
  rm mujoco.zip

RUN curl -o ~/miniconda.sh -O https://repo.anaconda.com/miniconda/Miniconda3-py37_4.12.0-Linux-x86_64.sh && \
  bash ~/miniconda.sh -b -p ~/conda && \
  rm ~/miniconda.sh

ENV PATH /home/$UNAME/conda/bin:$PATH

RUN conda install tensorflow-gpu

# Python packages.
RUN pip install --no-cache-dir \
  flatbuffers==1.12.0 \
  grpcio==1.32.0 \
  numpy==1.19.2 \
  six==1.15.0 \
  tensorflow-estimator==2.4.0 \
  typing-extensions==3.7.4 \
  wrapt==1.12.1 \
  gast==0.3.3 \
  google-auth==1.9.0 \
  opensimplex==0.3 \
  imageio==2.9.0 \
  imageio-ffmpeg \
  griddly \
# 'gym[atari]' \
#  atari_py \
  crafter \
  dm_control \
  ruamel.yaml \
  tensorflow_probability==0.12.2

RUN pip install --no-cache-dir gym==0.18.3 && pip install --no-cache-dir gym[atari] atari_py


# Atari ROMS.
RUN wget -L -nv http://www.atarimania.com/roms/Roms.rar && \
  unrar x Roms.rar && \
#  unzip ROMS.zip && \
  python -m atari_py.import_roms ROMS && \
  rm -rf Roms.rar ROMS.zip ROMS

# MuJoCo key.
ARG MUJOCO_KEY=""
RUN echo "$MUJOCO_KEY" > ~/.mujoco/mjkey.txt
RUN cat ~/.mujoco/mjkey.txt

# TF 2.4.1 cuda issue workaround
RUN mkdir -p /home/$UNAME/conda/bin/nvvm/libdevice && cp /home/$UNAME/conda/lib/libdevice.10.bc /home/$UNAME/conda/bin/nvvm/libdevice
ENV XLA_FLAGS --xla_gpu_cuda_data_dir=/home/$UNAME/conda/bin

# DreamerV2.
ENV TF_XLA_FLAGS --tf_xla_auto_jit=2
#COPY --chown=$UNAME . /app
CMD [ \
  "python", "dreamerv2/train.py", \
  "--logdir", "/logdir/$(date +%Y%m%d-%H%M%S)", \
  "--configs", "defaults", "atari", \
  "--task", "atari_pong" \
]