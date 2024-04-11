ARG UBUNTU_VERSION
ARG PYTHON_VERSION
ARG CUDA_RUNTIME

FROM nvidia/cuda:${CUDA_RUNTIME}-runtime-ubuntu${UBUNTU_VERSION}
LABEL maintainer=NDT email=<trongnd02@gmail.com>

#ARG DEBIAN_FRONTEND=noninteractive

ADD main.py requirements.txt /face_attribute/
ADD /src /face_attribute/src
WORKDIR /face_attribute

RUN apt update -y && apt upgrade -y
RUN apt-get install -y software-properties-common wget

RUN <<EOF
    apt install -y python${PYTHON_VERSION}
    add-apt-repository -y ppa:deadsnakes/ppa -y && apt update -y

    # Designate python version
    apt install -y python3-pip python${PYTHON_VERSION}-distutils
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1

    # Install dependencies
    pip install --upgrade pip
    pip install -r requirements.txt
    # Remove cache
    apt-get clean
EOF
# Run command
# docker run --rm -it --name test_cuda --gpus all --ipc host -v /home/trong/Downloads/Local/Source/python/semester_6/face_attribute/configs:/configs -v /home/trong/Downloads/Dataset:/Dataset python3 main.py  --config /configs/alexnet_binary_class.json
