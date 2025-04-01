from PIL import Image
from torchvision import transforms

def preprocess_image(image: Image.Image):
    # Define preprocessing pipeline_
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # MobileNetV3 typically uses 224x224 images
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return preprocess(image).unsqueeze(0)  # Add batch dimension
