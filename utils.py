import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

def load_model(model_path):
    """
    Load a PyTorch model from a checkpoint
    
    Args:
        model_path (str): Path to the model checkpoint
    
    Returns:
        torch.nn.Module: Loaded model
    """
    try:
        model = torch.load(model_path)
        model.eval()  # Set model to evaluation mode
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def preprocess_image(image_file):
    """
    Preprocess input image for model inference
    
    Args:
        image_file: Uploaded image file
    
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    # Open the image
    image = Image.open(image_file).convert('RGB')
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Adjust size to match model input
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # Standard ImageNet normalization
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Apply transforms
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return input_tensor

def postprocess_mask(mask_tensor):
    """
    Convert model output mask to a format that can be sent back
    
    Args:
        mask_tensor (torch.Tensor): Model output mask
    
    Returns:
        numpy.ndarray: Processed mask as numpy array
    """
    # Convert to numpy and remove batch dimension
    mask = mask_tensor.detach().squeeze().numpy()
    
    # Normalize mask to 0-255 range for easier display
    mask = (mask - mask.min()) / (mask.max() - mask.min()) * 255
    mask = mask.astype(np.uint8)
    
    return mask