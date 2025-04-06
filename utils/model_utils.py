from PIL import Image
from torchvision import transforms
import torch
import numpy as np

def preprocess_image(image: Image.Image):
    # """
    # Preprocess an image for inference with the LeafSense model.
    # Uses the exact same normalization as during training.
    # """
    # # Ensure the image is in RGB mode
    # if image.mode != "RGB":
    #     image = image.convert("RGB")
    
    # # Define preprocessing pipeline - EXACTLY match validation transforms from training
    # preprocess = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    
    # # Apply transformations and add batch dimension
    # tensor = preprocess(image).unsqueeze(0)

    print("🖼️ Preprocessing input image...")

    if image.mode != "RGB":
        image = image.convert("RGB")
        print("🔄 Converted image to RGB.")

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    tensor = preprocess(image).unsqueeze(0)
    print(f"✅ Image tensor shape: {tensor.shape}")

    
    return tensor

def get_prediction_info(output_tensor, top_k=3):
    """
    Process model output tensor to get class predictions and confidence scores
    """
    # Apply softmax to get probabilities
    probabilities = torch.nn.functional.softmax(output_tensor, dim=1)[0]
    
    # Get top-k confidence scores and class indices
    confidences, class_indices = torch.topk(probabilities, k=top_k)
    
    # Convert to Python lists
    confidences = [float(conf) * 100 for conf in confidences.tolist()]
    class_indices = class_indices.tolist()
    
    return {
        "top_classes": class_indices,
        "confidences": confidences
    }
