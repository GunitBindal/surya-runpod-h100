#!/bin/bash
set -e

# Configuration
DOCKER_USERNAME="gbdevelopers"
IMAGE_NAME="suryaocr-h100"
TAG="latest"
FULL_IMAGE="${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG}"

echo "=================================================="
echo "Building SuryaOCR Docker Image with Pre-cached Models"
echo "=================================================="
echo "Image: ${FULL_IMAGE}"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker Desktop."
    exit 1
fi

# Build the image
echo "Step 1: Building Docker image..."
docker build --platform linux/amd64 -t ${FULL_IMAGE} .

echo ""
echo "Step 2: Logging in to Docker Hub..."
docker login

echo ""
echo "Step 3: Pushing image to Docker Hub..."
docker push ${FULL_IMAGE}

echo ""
echo "=================================================="
echo "✓ Image built and pushed successfully!"
echo "=================================================="
echo ""
echo "Image: ${FULL_IMAGE}"
echo ""
echo "Next steps:"
echo "1. Go to RunPod Console → Serverless → Templates"
echo "2. Create new template with:"
echo "   - Container Image: ${FULL_IMAGE}"
echo "   - Docker Command: (leave empty, uses CMD from Dockerfile)"
echo "3. Deploy endpoint with H100 GPU"
echo "4. Set Active Workers = 1"
echo ""
echo "Benefits:"
echo "✓ Models pre-downloaded (~500MB cached in image)"
echo "✓ No pip install on every start"
echo "✓ Instant cold starts (~2-3 seconds)"
echo "✓ Ready for production workloads"
echo "=================================================="
