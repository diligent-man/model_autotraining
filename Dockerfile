FROM nvidia/cuda:12.3.2-runtime-ubuntu22.04
LABEL maintainer=NDT email=<trongnd02@gmail.com>

ADD main.py /face_attribute/
ADD requirements.txt /face_attribute/
ADD /src /face_attribute/src
WORKDIR /face_attribute

ARG PYTHON_VERSION=3.8

RUN apt update -y && upgrade -y
RUN apt-get install software-properties-common -y
RUN	add-apt-repository ppa:deadsnakes/ppa -y
RUN apt update -y
RUN apt install python${PYTHON_VERSION} -y
RUN apt install python3-pip -y
RUN apt-get clean
RUN pip install -r requirements.txt

# ENTRYPOINT ["/face_attribute/main.py"]


# docker run --rm -it --name test_cuda --gpus all --ipc host \
# -v /home/trong/Downloads/Local/Source/python/semester_6/face_attribute/configs:/configs \
# -v /home/trong/Downloads/Dataset:/Dataset

#python3 main.py --config /configs/alexnet_binary_class.json --dataset /dataset/small_celeb_A