import runpod
import base64
import io
import sys
from PIL import Image

print("Starting SuryaOCR Handler...", flush=True)

MODEL = None

def initialize_model():
    global MODEL
    if MODEL is None:
        print("Loading SuryaOCR model... This takes 60-90 seconds.", flush=True)
        try:
            from surya.ocr import OCRPredictor
            MODEL = OCRPredictor()
            print("✓ Model loaded successfully!", flush=True)
        except Exception as e:
            print(f"✗ Model loading failed: {e}", flush=True)
            raise
    return MODEL

def handler(job):
    print(f"Received job: {job.get('id', 'unknown')}", flush=True)
    
    try:
        # Initialize model on first request
        model = initialize_model()

        # Get input
        job_input = job.get("input", {})
        images_b64 = job_input.get("images", [])
        languages = job_input.get("languages", ["en"])

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

        # Run OCR
        print(f"Processing {len(images)} image(s) with languages: {languages}", flush=True)
        predictions = model(images, languages)
        print(f"✓ OCR completed", flush=True)

        # Format results
        results = []
        for pred in predictions:
            results.append({
                "text": [line.text for line in pred.text_lines],
                "boxes": [line.bbox for line in pred.text_lines]
            })

        return {"success": True, "results": results}

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"✗ Handler error: {e}\n{error_trace}", flush=True)
        return {"success": False, "error": str(e), "traceback": error_trace}

print("Starting RunPod serverless handler...", flush=True)
runpod.serverless.start({"handler": handler})