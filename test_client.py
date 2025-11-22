"""
Test client for SuryaOCR RunPod Endpoint
"""

import runpod
import base64
import json
import time
from pathlib import Path

# Set your RunPod API key
runpod.api_key = "your_runpod_api_key_here"

# Your endpoint ID (you'll get this after deployment)
ENDPOINT_ID = "your_endpoint_id_here"


def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def test_ocr_single_image(image_path: str):
    """Test OCR on a single image"""
    print(f"\n=== Testing OCR on {image_path} ===")

    # Encode image
    image_b64 = encode_image_to_base64(image_path)

    # Create endpoint
    endpoint = runpod.Endpoint(ENDPOINT_ID)

    # Run inference
    start_time = time.time()

    run_request = endpoint.run({
        "input": {
            "images": [image_b64],
            "operation": "ocr",
            "languages": ["en"]
        }
    })

    # Wait for result
    result = run_request.output()

    elapsed = time.time() - start_time

    print(f"\nProcessing time: {elapsed:.2f} seconds")
    print(f"\nResults:")
    print(json.dumps(result, indent=2))

    return result


def test_batch_ocr(image_paths: list):
    """Test OCR on multiple images (batched)"""
    print(f"\n=== Testing Batch OCR on {len(image_paths)} images ===")

    # Encode all images
    images_b64 = [encode_image_to_base64(path) for path in image_paths]

    # Create endpoint
    endpoint = runpod.Endpoint(ENDPOINT_ID)

    # Run inference
    start_time = time.time()

    run_request = endpoint.run({
        "input": {
            "images": images_b64,
            "operation": "ocr",
            "languages": ["en"]
        }
    })

    # Wait for result
    result = run_request.output()

    elapsed = time.time() - start_time

    print(f"\nProcessing time: {elapsed:.2f} seconds")
    print(f"Average per image: {elapsed/len(image_paths):.2f} seconds")
    print(f"\nResults:")
    print(json.dumps(result, indent=2))

    return result


def test_full_pipeline(image_path: str):
    """Test full OCR pipeline (detection, layout, OCR, tables, reading order)"""
    print(f"\n=== Testing Full Pipeline on {image_path} ===")

    # Encode image
    image_b64 = encode_image_to_base64(image_path)

    # Create endpoint
    endpoint = runpod.Endpoint(ENDPOINT_ID)

    # Run inference
    start_time = time.time()

    run_request = endpoint.run({
        "input": {
            "images": [image_b64],
            "operation": "full",
            "languages": ["en"]
        }
    })

    # Wait for result
    result = run_request.output()

    elapsed = time.time() - start_time

    print(f"\nProcessing time: {elapsed:.2f} seconds")
    print(f"\nResults:")
    print(json.dumps(result, indent=2))

    return result


def test_layout_analysis(image_path: str):
    """Test layout analysis only"""
    print(f"\n=== Testing Layout Analysis on {image_path} ===")

    # Encode image
    image_b64 = encode_image_to_base64(image_path)

    # Create endpoint
    endpoint = runpod.Endpoint(ENDPOINT_ID)

    # Run inference
    start_time = time.time()

    run_request = endpoint.run({
        "input": {
            "images": [image_b64],
            "operation": "layout"
        }
    })

    # Wait for result
    result = run_request.output()

    elapsed = time.time() - start_time

    print(f"\nProcessing time: {elapsed:.2f} seconds")
    print(f"\nResults:")
    print(json.dumps(result, indent=2))

    return result


def test_table_recognition(image_path: str):
    """Test table recognition only"""
    print(f"\n=== Testing Table Recognition on {image_path} ===")

    # Encode image
    image_b64 = encode_image_to_base64(image_path)

    # Create endpoint
    endpoint = runpod.Endpoint(ENDPOINT_ID)

    # Run inference
    start_time = time.time()

    run_request = endpoint.run({
        "input": {
            "images": [image_b64],
            "operation": "table"
        }
    })

    # Wait for result
    result = run_request.output()

    elapsed = time.time() - start_time

    print(f"\nProcessing time: {elapsed:.2f} seconds")
    print(f"\nResults:")
    print(json.dumps(result, indent=2))

    return result


if __name__ == "__main__":
    # Example usage - replace with your actual image paths
    print("SuryaOCR RunPod Test Client")
    print("=" * 50)

    # Test single image OCR
    # test_ocr_single_image("path/to/your/image.jpg")

    # Test batch OCR
    # test_batch_ocr([
    #     "path/to/image1.jpg",
    #     "path/to/image2.jpg",
    #     "path/to/image3.jpg"
    # ])

    # Test full pipeline
    # test_full_pipeline("path/to/your/document.jpg")

    # Test layout analysis
    # test_layout_analysis("path/to/your/document.jpg")

    # Test table recognition
    # test_table_recognition("path/to/your/table.jpg")

    print("\nTo use this client:")
    print("1. Set RUNPOD_API_KEY and ENDPOINT_ID at the top of the file")
    print("2. Uncomment the test you want to run")
    print("3. Replace image paths with your actual images")
