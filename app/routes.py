from flask import Blueprint, request, jsonify
from PIL import Image
import io
import torch
import json
import base64
import os
from mobilenet_v3_large_weights.mobilenet_v3 import load_model
from utils.model_utils import preprocess_image, get_prediction_info

main = Blueprint('main', __name__)

# Add this near the beginning of your routes.py file, after defining 'main'

# @main.route("/", methods=["GET"])
# def index():
#     return jsonify({
#         "status": "online",
#         "message": "LeafSense API server is running",
#         "endpoints": {
#             "POST /predict": "Image classification endpoint"
#         }
#     })

# Load the model at the start to avoid reloading on each request
model = load_model("mobilenet_v3_large_weights/mobilenetv3_best_accuracy7.pth")  # Path to your model file

# Load leaf data from JSON file
with open("data.json", "r") as f:
    leaf_data = json.load(f)

# Create a mapping from class IDs to leaf information
class_to_leaf_info = {leaf["id"]: leaf for leaf in leaf_data}

# Function to read image file and convert to base64
def get_image_base64(image_path):
    try:
        # Check if file exists
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return None
            
        # Read image file and convert to base64
        with open(image_path, "rb") as img_file:
            img_data = img_file.read()
            encoded = base64.b64encode(img_data).decode('utf-8')
            return encoded
    except Exception as e:
        print(f"Error reading image file: {e}")
        return None

# Prediction endpoint
@main.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    # Handle OPTIONS request (preflight) explicitly
    if request.method == "OPTIONS":
        print("Handling OPTIONS request")
        return "", 204
        
    # Check for JSON data with base64 image
    if not request.json or 'image' not in request.json:
        print("No image data in request")
        return jsonify({"error": "No image data provided"}), 400

    try:
        # Get base64 image data from request
        image_data = request.json['image']
        
        # Handle data URL format if present
        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]
            
        # Log request data size for debugging
        print(f"Image data size: {len(image_data) / 1024:.2f} KB")
        
        # Decode base64 to bytes
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Log image details for debugging
        print(f"Image size: {image.size}, mode: {image.mode}")
        print(f"Image format: {image.format}")
        
        # Show EXIF data if available
        if hasattr(image, '_getexif') and image._getexif() is not None:
            print("EXIF data present in image")
        
        input_tensor = preprocess_image(image)
        # Print tensor shape and values for debugging
        print(f"Input tensor shape: {input_tensor.shape}")
        print(f"Input tensor min: {input_tensor.min().item()}, max: {input_tensor.max().item()}")

        # Perform the prediction
        with torch.no_grad():
            output = model(input_tensor)
            
            # Get prediction details
            prediction_info = get_prediction_info(output, top_k=5)
            predicted_class = prediction_info["top_classes"][0]
            confidence = prediction_info["confidences"][0] / 100  # Convert back to 0-1 scale
            
            print(f"Top-5 classes: {prediction_info['top_classes']}")
            print(f"Top-5 confidences: {[f'{conf:.2f}%' for conf in prediction_info['confidences']]}")
            
            # Also print all confidence scores for debugging
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
            print("All confidence scores:")
            for idx, prob in enumerate(probabilities.tolist()):
                print(f"Class {idx}: {prob:.4f} ({prob*100:.2f}%)")

        # Get leaf information
        leaf_info = class_to_leaf_info.get(predicted_class, {"error": "Class ID not found"})
        leaf_info = leaf_info.copy()
        leaf_info["confidence"] = confidence
        leaf_info["all_predictions"] = {
            "classes": prediction_info["top_classes"],
            "confidences": prediction_info["confidences"]
        }
        
        # Add image data if available
        if "imagePath" in leaf_info:
            image_path = leaf_info["imagePath"]
            # If path is relative, assume it's relative to the static folder
            if not os.path.isabs(image_path):
                image_path = os.path.join(image_path)
                
            image_base64 = get_image_base64(image_path)
            if image_base64:
                leaf_info["imageData"] = image_base64
                leaf_info["imageType"] = os.path.splitext(image_path)[1][1:].lower()  # Get extension without dot
            else:
                leaf_info["imageError"] = "Image not found or could not be read"

        return jsonify(leaf_info)

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500 
        # return jsonify({"error": "Prediction failed"}), 500