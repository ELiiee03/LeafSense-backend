from flask import Blueprint, request, jsonify
from PIL import Image
import io
import torch
import json
from mobilenet_v3_small_weights.mobilenet_v3 import load_model
from utils.model_utils import preprocess_image

main = Blueprint('main', __name__)

# Load the model at the start to avoid reloading on each request
model = load_model("mobilenet_v3_small_weights/mobilenetv3_weights.pth")  # Path to your model file

# Load leaf data from JSON file
with open("data.json", "r") as f:
    leaf_data = json.load(f)

# Create a mapping from class IDs to leaf information
class_to_leaf_info = {leaf["id"]: leaf for leaf in leaf_data}

# Prediction endpoint
@main.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        print("No file part in request.files")
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == '':
        print("No selected file")
        return jsonify({"error": "No file provided"}), 400

    try:
        # Open the image file and preprocess it
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        input_tensor = preprocess_image(image)

        # Perform the prediction
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            predicted_class = predicted_class.item()  # Extract the predicted class
            confidence = confidence.item()  # Extract the confidence score

        # Get leaf information based on the predicted class
        leaf_info = class_to_leaf_info.get(predicted_class, {"error": "Class ID not found"})
        
        # Ensure leaf_info is a copy to avoid modifying the original data
        leaf_info = leaf_info.copy()
        leaf_info["confidence"] = confidence  # Add confidence score to the response

        # Return the result in JSON format
        return jsonify(leaf_info)

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": "Prediction failed"}), 500