# SuryaOCR RunPod Serverless - Optimized for Maximum Speed
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# Set environment variables for maximum H100 performance
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    # CUDA optimizations
    CUDA_MODULE_LOADING=LAZY \
    TORCH_CUDA_ARCH_LIST="9.0" \
    NVIDIA_TF32_OVERRIDE=1 \
    TORCH_CUDNN_V8_API_ENABLED=1 \
    # Aggressive batch sizes for H100 80GB
    RECOGNITION_BATCH_SIZE=1024 \
    DETECTOR_BATCH_SIZE=128 \
    # Memory optimizations
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True \
    # Enable JIT compilation
    PYTORCH_JIT=1 \
    # Threading
    OMP_NUM_THREADS=16 \
    MKL_NUM_THREADS=16 \
    # Disable profiling overhead
    CUDA_LAUNCH_BLOCKING=0 \
    # Enable cuDNN benchmarking
    TORCH_CUDNN_BENCHMARK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies with optimizations
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    surya-ocr==0.17.0 \
    runpod==1.8.1 \
    pillow-simd==10.4.0 || pip install --no-cache-dir pillow==10.4.0 && \
    pip install --no-cache-dir \
    torch-tensorrt \
    nvidia-cudnn-cu12

# Set working directory
WORKDIR /app

# Copy handler
COPY handler_final.py /app/handler.py

# Pre-download and pre-compile ALL Surya models
RUN python3 << 'EOF'
import torch
from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor

print('Downloading and compiling models...')

# Enable optimizations
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# Load models
print('Loading Foundation model...')
fp = FoundationPredictor()
print('Foundation model cached')

print('Loading Recognition model...')
rp = RecognitionPredictor(fp)
print('Recognition model cached')

print('Loading Detection model...')
dp = DetectionPredictor()
print('Detection model cached')

# Pre-warm models with dummy data (compiles kernels)
print('Pre-warming models...')
from PIL import Image
import numpy as np

# Create dummy image
dummy_img = Image.fromarray(np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8))

try:
    # Run inference once to compile CUDA kernels
    _ = rp([dummy_img], det_predictor=dp)
    print('Models pre-warmed successfully!')
except Exception as e:
    print(f'Pre-warming warning (non-critical): {e}')

print('All optimizations complete!')
EOF

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import runpod; print('healthy')" || exit 1

# Start handler
CMD ["python3", "-u", "/app/handler.py"]