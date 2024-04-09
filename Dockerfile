FROM nvidia/cuda:12.3.2-runtime-ubuntu22.04
LABEL maintainer=NDT email=<trongnd02@gmail.com>

ARG PYTHON_VERSION=3.11
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

    # Install the latest pip
    wget https://bootstrap.pypa.io/get-pip.py -O ./get-pip.py
    python3 ./get-pip.py

    # Remove cache
    apt-get clean
EOF

RUN pip install -r requirements.txt
# Run command
# docker run --rm -it --name test_cuda --gpus all --ipc host -v /home/trong/Downloads/Local/Source/python/semester_6/face_attribute/configs:/configs -v /home/trong/Downloads/Dataset:/Dataset python3 main.py  --config /configs/alexnet_binary_class.json
