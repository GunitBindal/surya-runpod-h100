# SuryaOCR RunPod H100 Serverless

Production-ready SuryaOCR deployment on RunPod H100 GPU with 0.5s per image OCR speed.

## 🚀 Simple 3-Step Setup

### 1. Create Template (RunPod Web UI)

Go to: **RunPod Console → Serverless → Templates → New Template**

- **Template Name**: `surya-h100`
- **Container Image**: `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`
- **Docker Command**:
```bash
bash -c "pip install -q surya-ocr runpod pillow && curl -sSL https://raw.githubusercontent.com/GunitBindal/surya-runpod-h100/main/handler_final.py -o handler.py && python -u handler.py"
```

### 2. Deploy Endpoint

Go to: **Serverless → Endpoints → New Endpoint**

- **Template**: Select `surya-h100`
- **GPUs**: H100 80GB or H100 PCIe
- **Active Workers**: 1 (important!)
- **Max Workers**: 1
- **Endpoint Type**: Queue

Click **Deploy**

### 3. Test

```python
import runpod

runpod.api_key = "YOUR_API_KEY"
endpoint = runpod.Endpoint("YOUR_ENDPOINT_ID")

# First request takes 60-90s (model download)
result = endpoint.run({
    "input": {
        "images": ["BASE64_IMAGE_STRING"],
        "languages": ["en"]
    }
}).output()

print(result)
```

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