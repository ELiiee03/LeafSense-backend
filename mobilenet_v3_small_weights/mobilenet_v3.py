import torch
from torchvision.models import mobilenet_v3_small  # Or large, depending on your model

def load_model(model_path="mobilenetv3.pth"):
    # Load a pre-trained MobileNetV3 model (adjust based on your model specifics)
    model = mobilenet_v3_small(pretrained=False)  # Set pretrained=False for custom models
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model
