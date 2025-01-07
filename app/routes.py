from flask import Blueprint, request, jsonify
from PIL import Image
import io
import torch
from mobilenet_v3_small_weights.mobilenet_v3 import load_model
from utils.model_utils import preprocess_image

main = Blueprint('main', __name__)

# Load the model at the start to avoid reloading on each request
model = load_model("mobilenet_v3_small_weights/leaf_classification_model.pkl")  # Path to your model file

# Prediction endpoint
@main.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    try:
        # Open the image file and preprocess it
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        input_tensor = preprocess_image(image)

        # Perform the prediction
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted_class = torch.max(output, 1)
            predicted_class = predicted_class.item()  # Extract the predicted class

        # Return the result in JSON format
        return jsonify({"class_id": predicted_class})

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": "Prediction failed"}), 500