from flask import Blueprint, request, jsonify
from PIL import Image
import io
import torch
import json
import base64
import os
import traceback
import psutil  # For memory logging
import gc
import time

from mobilenet_v3_large_weights.mobilenet_v3 import load_model
from utils.model_utils import preprocess_image, get_prediction_info

main = Blueprint('main', __name__)

# Log initial memory usage at server startup
print("📊 MEMORY: Initial server startup")
process = psutil.Process(os.getpid())
print(f"📊 MEMORY: RSS: {process.memory_info().rss / (1024 * 1024):.2f} MB")
print(f"📊 MEMORY: VMS: {process.memory_info().vms / (1024 * 1024):.2f} MB")

# Load the model at the start to avoid reloading on each request
print("🚀 Loading model at server startup...")
model = load_model("mobilenet_v3_large_weights/mobilenetv3_best_accuracy7.pth")
print("✅ Model loaded successfully.")

# Log memory after model load
print("📊 MEMORY: After model load")
print(f"📊 MEMORY: RSS: {process.memory_info().rss / (1024 * 1024):.2f} MB")
print(f"📊 MEMORY: VMS: {process.memory_info().vms / (1024 * 1024):.2f} MB")

# Load leaf data from JSON file
with open("data.json", "r") as f:
    leaf_data = json.load(f)

print("📄 Leaf class data loaded.")
class_to_leaf_info = {leaf["id"]: leaf for leaf in leaf_data}

# Utility to read image and return base64
def get_image_base64(image_path):
    try:
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return None
        with open(image_path, "rb") as img_file:
            img_data = img_file.read()
            encoded = base64.b64encode(img_data).decode('utf-8')
            return encoded
    except Exception as e:
        print(f"❌ Error reading image file: {e}")
        return None

# Enhanced memory logging utility
def log_memory_usage(label=""):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    rss_mb = mem_info.rss / (1024 * 1024)
    vms_mb = mem_info.vms / (1024 * 1024)
    
    # Get Python-specific memory info
    py_mem = {}
    try:
        import sys
        py_mem['objects'] = len(gc.get_objects())
        py_mem['refs'] = len(gc.get_referrers(*gc.get_objects()[:1]))
    except Exception as e:
        py_mem['error'] = str(e)
    
    # Log detailed memory info
    print(f"📊 MEMORY [{label}]")
    print(f"📊 MEMORY: RSS: {rss_mb:.2f} MB")
    print(f"📊 MEMORY: VMS: {vms_mb:.2f} MB")
    print(f"📊 MEMORY: Python objects: {py_mem.get('objects', 'N/A')}")
    print(f"📊 MEMORY: GC counts: {gc.get_count()}")
    
    # Critical warning if approaching Render's limit (512MB)
    if rss_mb > 450:
        print(f"⚠️ CRITICAL: Memory usage ({rss_mb:.2f} MB) approaching Render's 512MB limit!")
    
    return rss_mb

# Log memory after complete initialization
log_memory_usage("Server fully initialized")

# Prediction route
@main.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    request_id = f"req-{int(time.time() * 1000)}"
    print(f"\n=== Starting request {request_id} ===")
    initial_memory = log_memory_usage(f"Request start {request_id}")
    
    if request.method == "OPTIONS":
        print("⚙️ Handling OPTIONS preflight request")
        return "", 204

    if not request.json or 'image' not in request.json:
        print("❌ No image data in request")
        return jsonify({"error": "No image data provided"}), 400

    try:
        # Extract image data
        image_data = request.json['image']
        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]

        data_size_kb = len(image_data) / 1024
        print(f"📦 Received image data: {data_size_kb:.2f} KB")
        
        # Track memory after receiving data
        log_memory_usage(f"After receiving image data ({data_size_kb:.2f} KB)")
        
        # Decode image
        image_bytes = base64.b64decode(image_data)
        log_memory_usage("After base64 decode")
        
        # Clear image_data to free memory
        image_data = None
        
        # Open image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Clear image_bytes to free memory
        image_bytes = None
        
        print(f"🖼️ Image size: {image.size}, mode: {image.mode}, format: {image.format}")
        log_memory_usage("After image open")
        
        # Process image 
        input_tensor = preprocess_image(image)
        
        # Clear original image to free memory
        image = None
        
        print(f"📥 Input tensor shape: {input_tensor.shape}")
        log_memory_usage("After preprocessing (tensor creation)")

        # Run garbage collection before inference
        collected = gc.collect()
        print(f"🧹 GC collected {collected} objects")

        try:
            with torch.no_grad():
                print(f"⚡ Running model inference for {request_id}...")
                log_memory_usage("Right before model inference")
                
                # Run inference
                output = model(input_tensor)
                
                log_memory_usage("Immediately after model inference")
                print(f"✅ Model inference complete for {request_id}.")
                
                # Clear input tensor to free memory
                input_tensor = None
                
                # Run garbage collection
                gc.collect()
        except Exception as model_err:
            print("❌ Model inference failed:")
            traceback.print_exc()
            log_memory_usage("After model failure")
            return jsonify({"error": "Model inference failed", "details": str(model_err)}), 500

        # Process results
        prediction_info = get_prediction_info(output, top_k=5)
        predicted_class = prediction_info["top_classes"][0]
        confidence = prediction_info["confidences"][0] / 100
        
        # Clear output to free memory
        output = None
        
        log_memory_usage("After prediction processing")

        print(f"🎯 Top-5 classes: {prediction_info['top_classes']}")
        print(f"📈 Top-5 confidences: {[f'{conf:.2f}%' for conf in prediction_info['confidences']]}")

        probabilities = torch.nn.functional.softmax(output, dim=1)[0] if output is not None else None
        if probabilities is not None:
            print("📊 All confidence scores:")
            for idx, prob in enumerate(probabilities.tolist()):
                print(f"  Class {idx}: {prob:.4f} ({prob*100:.2f}%)")
            
            # Clear probabilities to free memory
            probabilities = None

        # Build response
        leaf_info = class_to_leaf_info.get(predicted_class, {"error": "Class ID not found"})
        leaf_info = leaf_info.copy()
        leaf_info["confidence"] = confidence
        leaf_info["all_predictions"] = {
            "classes": prediction_info["top_classes"],
            "confidences": prediction_info["confidences"]
        }

        # Handle leaf image
        log_memory_usage("Before handling leaf reference image")
        if "imagePath" in leaf_info:
            image_path = leaf_info["imagePath"]
            if not os.path.isabs(image_path):
                image_path = os.path.join(image_path)

            image_base64 = get_image_base64(image_path)
            log_memory_usage("After encoding reference image")
            
            if image_base64:
                leaf_info["imageData"] = image_base64
                leaf_info["imageType"] = os.path.splitext(image_path)[1][1:].lower()
            else:
                leaf_info["imageError"] = "Image not found or could not be read"

        # Run final garbage collection
        collected = gc.collect()
        print(f"🧹 Final GC collected {collected} objects")
        
        # Log final memory
        final_memory = log_memory_usage(f"End of request {request_id}")
        print(f"📊 MEMORY: Delta for this request: {final_memory - initial_memory:.2f} MB")
        print(f"=== Completed request {request_id} ===\n")
        
        return jsonify(leaf_info)

    except Exception as e:
        print(f"❌ Error during prediction for {request_id}:")
        traceback.print_exc()
        log_memory_usage("After error")
        return jsonify({"error": str(e)}), 500
