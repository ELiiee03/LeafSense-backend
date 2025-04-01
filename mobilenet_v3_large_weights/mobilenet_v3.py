import torch
from torchvision.models import mobilenet_v3_large

def load_model(model_path="mobilenet_v3_small_weights/mobilenetv3_weights.pth"):
    # Load a pre-trained MobileNetV3 model
    model = mobilenet_v3_large(pretrained=False)
    
    # Modify the classifier layer to match the number of classes in your dataset
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 10)  # Adjust the number of output units to 4
    
    # Load the custom weights
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model
