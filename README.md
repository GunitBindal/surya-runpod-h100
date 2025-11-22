# SuryaOCR RunPod H100 Serverless

Production-ready SuryaOCR deployment on RunPod H100 GPU with 0.5s per image OCR speed.

## 🚀 Quick Setup (2 Options)

### Option 1: Pre-built Docker Image (RECOMMENDED - Instant Startup)

**Best for production:** Models are pre-downloaded, no cold start delays!

1. **Build and push custom image:**
   ```bash
   ./build_and_push.sh
   ```
   This will:
   - Build Docker image with all models pre-cached (~3GB)
   - Push to Docker Hub (gbdevelopers/suryaocr-h100:latest)
   - Models download during build, NOT at runtime

2. **Create RunPod Template:**
   - Go to: RunPod Console → Serverless → Templates → New Template
   - Container Image: `gbdevelopers/suryaocr-h100:latest`
   - Docker Command: (leave empty)
   - GPU: H100 80GB or H100 PCIe

3. **Deploy Endpoint:**
   - Template: Select your template
   - Active Workers: 1
   - Max Workers: 1
   - Endpoint Type: Queue

**Benefits:**
- ✅ Instant cold starts (2-3 seconds)
- ✅ No model download wait time
- ✅ No pip installs on startup
- ✅ Production-ready

### Option 2: GitHub Handler (Simple but Slower)

**Good for testing:** Downloads on every worker start.

1. **Create Template:**
   - Container Image: `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`
   - Docker Command:
   ```bash
   bash -c "pip install --no-cache-dir surya-ocr runpod pillow && curl -sSL https://raw.githubusercontent.com/GunitBindal/surya-runpod-h100/main/handler_final.py -o handler.py && python -u handler.py"
   ```

2. **Deploy Endpoint** (same as Option 1, step 3)

**Drawbacks:**
- ⏱️ First request: 60-90 seconds (downloads models)
- 📦 Installs packages every worker start
- 💰 Wastes compute time on setup

## 📝 Usage

## 📋 Key Points

- **First request**: 60-90 seconds (downloads Surya models ~500MB)
- **Subsequent requests**: ~0.5 seconds per image
- **Active Workers = 1**: Keeps worker warm, prevents queue issues
- **Logs visible**: Handler prints status to worker logs

## 📁 Files

- `handler_final.py` - Optimized handler with logging
- `docker_command.txt` - RunPod Docker command
- `test_client.py` - Python test client

## 🔧 Troubleshooting

**Requests stuck IN_QUEUE?**
- Set Active Workers = 1 in endpoint settings
- Wait 90 seconds for first model download
- Check worker logs for errors

**Worker crashing?**
- Verify H100 GPU is selected
- Ensure Docker command is exactly as shown above
- Check logs for memory issues

## Current Status

✓ Handler updated with better logging  
✓ Simplified deployment (no Docker build)  
✓ Single-command setup via GitHub