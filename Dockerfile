# SuryaOCR RunPod Serverless - Pre-baked Image with Models
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# Set environment variables for optimal H100 performance
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_MODULE_LOADING=LAZY \
    TORCH_CUDA_ARCH_LIST="9.0" \
    NVIDIA_TF32_OVERRIDE=1 \
    RECOGNITION_BATCH_SIZE=512 \
    DETECTOR_BATCH_SIZE=64 \
    TORCH_CUDNN_V8_API_ENABLED=1 \
    OMP_NUM_THREADS=8 \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    surya-ocr==0.17.0 \
    runpod==1.8.1 \
    pillow==10.4.0

# Set working directory
WORKDIR /app

# Copy handler
COPY handler_final.py /app/handler.py

# Pre-download ALL Surya models at build time (this caches them in the image)
RUN python3 -c "\
from surya.foundation import FoundationPredictor; \
from surya.recognition import RecognitionPredictor; \
from surya.detection import DetectionPredictor; \
print('Downloading Foundation model...'); \
fp = FoundationPredictor(); \
print('Foundation model cached'); \
print('Downloading Recognition model...'); \
rp = RecognitionPredictor(fp); \
print('Recognition model cached'); \
print('Downloading Detection model...'); \
dp = DetectionPredictor(); \
print('Detection model cached'); \
print('All models successfully cached in image!')"

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import runpod; print('healthy')" || exit 1

# Start handler
CMD ["python3", "-u", "/app/handler.py"]