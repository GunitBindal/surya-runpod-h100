# SuryaOCR RunPod Serverless - Maximum Speed with Space Optimization
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
    # Threading optimizations
    OMP_NUM_THREADS=16 \
    MKL_NUM_THREADS=16 \
    # Disable profiling overhead
    CUDA_LAUNCH_BLOCKING=0 \
    # Enable cuDNN benchmarking for auto-tuning
    TORCH_CUDNN_BENCHMARK=1

# Install system dependencies and clean up in one layer
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install Python dependencies with cleanup
RUN pip install --no-cache-dir \
    surya-ocr==0.17.0 \
    runpod==1.8.1 && \
    pip install --no-cache-dir pillow-simd==10.4.0 || pip install --no-cache-dir pillow==10.4.0 && \
    rm -rf /root/.cache/pip /tmp/*

WORKDIR /app

# Copy handler
COPY handler_final.py /app/handler.py

# Pre-download models AND pre-warm CUDA kernels (with cleanup)
RUN python3 << 'EOF' && rm -rf /tmp/* /root/.cache/*
import torch
from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor

print('🚀 Optimizing for H100...')

# Enable all optimizations
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# Load models
print('📥 Downloading Foundation model...')
fp = FoundationPredictor()

print('📥 Downloading Recognition model...')
rp = RecognitionPredictor(fp)

print('📥 Downloading Detection model...')
dp = DetectionPredictor()

print('✅ All models cached!')

# Pre-warm with CPU dummy data (saves space, still compiles kernels)
print('🔥 Pre-warming CUDA kernels...')
from PIL import Image
import numpy as np

dummy = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))

try:
    # This compiles CUDA kernels at build time
    with torch.inference_mode():
        _ = rp([dummy], det_predictor=dp)
    print('✅ CUDA kernels pre-compiled!')
except:
    print('⚠️  Pre-warming skipped (will compile on first request)')

print('🎯 Optimization complete!')
EOF

# Final cleanup
RUN rm -rf /tmp/* /root/.cache/* /var/tmp/*

CMD ["python3", "-u", "/app/handler.py"]