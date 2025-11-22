# SuryaOCR RunPod H100 Serverless

Production-ready SuryaOCR deployment on RunPod H100 GPU.

## Quick Setup

1. Create RunPod template with image: `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`
2. Set Docker Command:
```bash
pip install -q surya-ocr runpod pillow && curl -o handler.py https://raw.githubusercontent.com/YOUR_USERNAME/surya-runpod-h100/main/handler_final.py && python handler.py
```
3. Deploy endpoint with H100 GPU

## Files

- `handler_final.py` - Production handler
- `docker_command.txt` - Docker start command
- `test_client.py` - Test client

## Usage

```python
import runpod
runpod.api_key = "YOUR_API_KEY"
endpoint = runpod.Endpoint("YOUR_ENDPOINT_ID")

result = endpoint.run({"input": {"images": [img_b64], "languages": ["en"]}}).output()
```
