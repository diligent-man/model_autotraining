declare -a PYTHON_VERSIONS=("3.8" "3.11")
declare -a UBUNTU_VERSIONS=("20.04" "22.04")

TAG="0.0.0" # default tag for image
CUDA_RUNTIME="12.4.1"
DOCKER_REGISTRY="trongnd02/ubuntu-cuda-py"


for UBUNTU_VERSION in "${UBUNTU_VERSIONS[@]}"; do
    for PYTHON_VERSION in "${PYTHON_VERSIONS[@]}"; do
        IMAGE_NAME="ubuntu${UBUNTU_VERSION}-cuda${CUDA_RUNTIME}-py${PYTHON_VERSION}"
        
        echo "##############################################################################"
        echo "         Containerize for ubuntu: ${UBUNTU_VERSION}, python: ${PYTHON_VERSION}"
        echo "##############################################################################"
        
        # Build image
        docker build -t $IMAGE_NAME:$TAG \
                     --build-arg CUDA_RUNTIME=$CUDA_RUNTIME \
                     --build-arg UBUNTU_VERSION=$UBUNTU_VERSION \
                     --build-arg PYTHON_VERSION=$PYTHON_VERSION \
                     .
        
        # Tagging        
        echo "$IMAGE_NAME"
        echo "$DOCKER_REGISTRY:$IMAGE_NAME"
        docker tag $IMAGE_NAME:$TAG $DOCKER_REGISTRY:$IMAGE_NAME
        
        # Push to docker registry
        docker push $DOCKER_REGISTRY:$IMAGE_NAME
        echo ""
        echo ""
    done
done

