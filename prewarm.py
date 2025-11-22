#!/usr/bin/env python3
"""Pre-warm Surya models and compile CUDA kernels at Docker build time"""
import torch
from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
from PIL import Image
import numpy as np

print('🚀 Optimizing for H100...', flush=True)

# Enable all optimizations
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# Load models
print('📥 Downloading Foundation model...', flush=True)
fp = FoundationPredictor()

print('📥 Downloading Recognition model...', flush=True)
rp = RecognitionPredictor(fp)

print('📥 Downloading Detection model...', flush=True)
dp = DetectionPredictor()

print('✅ All models cached!', flush=True)

# Pre-warm with dummy inference to compile CUDA kernels
print('🔥 Pre-warming CUDA kernels...', flush=True)
dummy = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))

try:
    with torch.inference_mode():
        _ = rp([dummy], det_predictor=dp)
    print('✅ CUDA kernels pre-compiled!', flush=True)
except Exception as e:
    print(f'⚠️  Pre-warming skipped: {e}', flush=True)
    print('(Kernels will compile on first request)', flush=True)

print('🎯 Optimization complete!', flush=True)
