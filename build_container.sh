DOCKER_REGISTRY="trongnd02/ubuntu-cuda-py"

declare -a UBUNTU_VERSIONS=("20.04" "22.04")
declare -a PYTHON_VERSIONS=("3.8" "3.11")
TAG = "0.0.0" # default tag for image

for UBUNTU_VERSION in "${UBUNTU_VERSIONS[@]}"; do
    for PYTHON_VERSION in "${PYTHON_VERSIONS[@]}"; do
        IMAGE_NAME="ubuntu${UBUNTU_VERSION}-cuda12.3.2-py${PYTHON_VERSION}"
        
        echo "##############################################################################"
        echo "         Containerize for ubuntu: ${UBUNTU_VERSION}, python: ${PYTHON_VERSION}"
        echo "##############################################################################"
        
        # Build image
        docker build -t $IMAGE_NAME:$TAG \
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

