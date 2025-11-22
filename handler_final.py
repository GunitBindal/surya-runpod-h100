import runpod
import base64
import io
from PIL import Image
from surya.ocr import OCRPredictor

MODEL = None

def handler(job):
    global MODEL

    # Initialize model on first request
    if MODEL is None:
        print("Loading SuryaOCR model...")
        MODEL = OCRPredictor()
        print("Model loaded!")

    try:
        # Get input
        job_input = job.get("input", {})
        images_b64 = job_input.get("images", [])
        languages = job_input.get("languages", ["en"])

        # Handle single image string
        if isinstance(images_b64, str):
            images_b64 = [images_b64]

        # Decode images
        images = []
        for img_b64 in images_b64:
            # Remove data URL prefix if present
            if img_b64.startswith("data:"):
                img_b64 = img_b64.split(",")[1]

            # Decode and convert
            img_bytes = base64.b64decode(img_b64)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            images.append(img)

        # Run OCR
        print(f"Processing {len(images)} image(s)...")
        predictions = MODEL(images, languages)

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
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

# Start the handler
runpod.serverless.start({"handler": handler})
