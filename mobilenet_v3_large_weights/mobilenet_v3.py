import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large
import os

def load_model(model_path="mobilenet_v3_large_weights/mobilenetv3_best_accuracy7.pth"): 
    """
    Loads a MobileNetV3 model with the LeafSense classifier architecture.
    This matches the exact architecture used during training in the notebook.
    """
    # Ensure path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}")
    
    print(f"Loading model from: {model_path}")
    
    # Create a mobilenet v3 large model with default structure
    model = mobilenet_v3_large(pretrained=False)
    
    # Modify the classifier to exactly match what was used during training
    in_features = model.classifier[0].in_features  # 960
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        
        nn.Linear(256, 10)  # 10 leaf classes
    )
    
    # Load the weights - weights were saved with 'model.' prefix
    state_dict = torch.load(model_path, map_location="cpu")
    
    # Remove 'model.' prefix from keys
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('model.'):
            new_key = key[6:]  # Remove 'model.' prefix
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    # Load weights
    model.load_state_dict(new_state_dict, strict=False)
    
    # Set to evaluation mode
    model.eval()
    
    return model

# import torch
# from torchvision.models import mobilenet_v3_large

# def load_model(model_path="mobilenet_v3_large_weights\mobilenetv3_best_accuracy7.pth"):
#     # Load a pre-trained MobileNetV3 model
#     model = mobilenet_v3_large(pretrained=False)
    
#     # Modify the classifier layer to match the number of classes in your dataset
#     model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 10)  # Adjust the number of output units to 4
    
#     # Load the custom weights
#     model.load_state_dict(torch.load(model_path, map_location="cpu"))
#     model.eval()
#     return model