import runpod
import base64
import io
import sys
import os
import torch
from PIL import Image

print("Starting SuryaOCR Handler...", flush=True)
print(f"ENV: RECOGNITION_BATCH_SIZE={os.getenv('RECOGNITION_BATCH_SIZE', 'not set')}", flush=True)
print(f"ENV: DETECTOR_BATCH_SIZE={os.getenv('DETECTOR_BATCH_SIZE', 'not set')}", flush=True)

# Enable PyTorch optimizations
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# Global model instances
FOUNDATION_PREDICTOR = None
RECOGNITION_PREDICTOR = None
DETECTION_PREDICTOR = None

def initialize_models():
    global FOUNDATION_PREDICTOR, RECOGNITION_PREDICTOR, DETECTION_PREDICTOR

    if FOUNDATION_PREDICTOR is None:
        print("Loading SuryaOCR models... This takes 2-3 seconds with pre-cached models.", flush=True)
        try:
            # Set batch sizes programmatically (fallback + override)
            from surya import settings
            settings.RECOGNITION_BATCH_SIZE = int(os.getenv('RECOGNITION_BATCH_SIZE', 1024))
            settings.DETECTOR_BATCH_SIZE = int(os.getenv('DETECTOR_BATCH_SIZE', 128))
            print(f"✓ Batch sizes set: RECOGNITION={settings.RECOGNITION_BATCH_SIZE}, DETECTOR={settings.DETECTOR_BATCH_SIZE}", flush=True)

            from surya.foundation import FoundationPredictor
            from surya.recognition import RecognitionPredictor
            from surya.detection import DetectionPredictor
            
            print("✓ Loading Foundation model...", flush=True)
            FOUNDATION_PREDICTOR = FoundationPredictor()
            
            print("✓ Loading Recognition model...", flush=True)
            RECOGNITION_PREDICTOR = RecognitionPredictor(FOUNDATION_PREDICTOR)
            
            print("✓ Loading Detection model...", flush=True)
            DETECTION_PREDICTOR = DetectionPredictor()
            
            print("✓ All models loaded successfully!", flush=True)
        except Exception as e:
            print(f"✗ Model loading failed: {e}", flush=True)
            raise
    
    return RECOGNITION_PREDICTOR, DETECTION_PREDICTOR

def handler(job):
    print(f"Received job: {job.get('id', 'unknown')}", flush=True)
    
    try:
        # Initialize models on first request
        recognition_predictor, detection_predictor = initialize_models()

        # Get input
        job_input = job.get("input", {})
        images_b64 = job_input.get("images", [])
        
        # Note: Surya auto-detects languages - no language parameter needed
        # The 'languages' input is accepted for API compatibility but not used

        if not images_b64:
            return {"success": False, "error": "No images provided"}

        # Handle single image string
        if isinstance(images_b64, str):
            images_b64 = [images_b64]

        # Decode images
        images = []
        for idx, img_b64 in enumerate(images_b64):
            try:
                # Remove data URL prefix if present
                if img_b64.startswith("data:"):
                    img_b64 = img_b64.split(",")[1]

                # Decode and convert
                img_bytes = base64.b64decode(img_b64)
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                images.append(img)
                print(f"✓ Image {idx+1} decoded: {img.size}", flush=True)
            except Exception as e:
                print(f"✗ Image {idx+1} decode failed: {e}", flush=True)
                return {"success": False, "error": f"Image {idx+1} decode failed: {str(e)}"}

        # Run OCR - Surya automatically detects languages
        # Batch sizes controlled by RECOGNITION_BATCH_SIZE env var in Dockerfile
        print(f"Processing {len(images)} image(s) with auto language detection", flush=True)
        predictions = recognition_predictor(
            images,
            det_predictor=detection_predictor
        )
        print(f"✓ OCR completed", flush=True)

        # Format results
        results = []
        for pred in predictions:
            text_lines = []
            for line in pred.text_lines:
                text_lines.append({
                    "text": line.text,
                    "confidence": line.confidence,
                    "bbox": line.bbox,
                    "polygon": line.polygon
                })
            
            results.append({
                "text_lines": text_lines,
                "page": getattr(pred, 'page', 0),
                "image_bbox": getattr(pred, 'image_bbox', None)
            })

        return {"success": True, "results": results}

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"✗ Handler error: {e}\n{error_trace}", flush=True)
        return {"success": False, "error": str(e), "traceback": error_trace}

print("Starting RunPod serverless handler...", flush=True)
runpod.serverless.start({"handler": handler})