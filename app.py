import cv2
import torch
import numpy as np
import base64
from flask import Flask, request, jsonify
from torchvision import transforms
import segmentation_models_pytorch as smp
from flask_cors import CORS


MODEL_PATH = 'models/best_model_checkpoint.pt'

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": ["http://localhost:3000", "https://advaithmalka.github.io/cop-classifier/"]}})
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    if request.method == 'OPTIONS':
        response.headers['Access-Control-Allow-Methods'] = 'DELETE, GET, POST, PUT'
        headers = request.headers.get('Access-Control-Request-Headers')
        if headers:
            response.headers['Access-Control-Allow-Headers'] = headers
    return response
app.after_request(add_cors_headers)

@app.route("/")
def index():
    return "<p>Welcome the Mito Detect API</p>"
# Model and device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define model architecture (ensure this matches your model's architecture)
def create_model(): 
    """
    Create and load the segmentation model
    
    Returns:
        torch.nn.Module: Loaded model
    """
    # Adjust these parameters to match your specific model configuration
    model = smp.Unet(
        encoder_name="resnet50",        # choose encoder, e.g. mobilenet_v2 or resnet34
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=1,                  # model input channels (1 for grayscale)
        classes=1,                      # model output channels (1 for binary segmentation)
    )
    
    # Load model weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model

# Load model when the application starts
model = create_model()

# Image preprocessing transforms
eval_transform = transforms.Compose([
    transforms.ToPILImage(),             # Convert numpy array to PIL image
    transforms.Resize((256, 256)),       # Resize to 256x256 or your desired size
    transforms.ToTensor(),               # Convert to tensor
])

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint for mitochondria mask prediction
    """
    print(request.files["file"])
    if 'file' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    # Read the uploaded image
    image_file = request.files['file']
    
    try:
        # Read image as grayscale and normalize
        img_array = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        img_array = img_array / 255.0  # Normalize to [0, 1]

        # Apply transformations
        test_img = eval_transform(img_array)

        # Add batch dimension and send to device
        test_img = test_img.unsqueeze(0).to(device)

        # Perform inference
        with torch.inference_mode():
            pred_mask = model(test_img).squeeze().cpu().numpy()
            pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255  # Threshold and scale to 0-255

        # Resize mask back to original image size
        pred_mask = cv2.resize(pred_mask, (img_array.shape[1], img_array.shape[0]))

        # === Create Overlay Image ===
        original_color = cv2.cvtColor((img_array * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)  # Convert grayscale to BGR


        # Generate random colors for each mitochondrion
        num_labels, labels = cv2.connectedComponents(pred_mask)
        color_mask = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)

        for i in range(1, num_labels):  # Skip background (label 0)
            color = np.random.randint(200, 255, size=(3,), dtype=np.uint8)  # Brighter colors
            color_mask[labels == i] = color  # Assign the random color

        overlay = cv2.addWeighted(original_color, 1.0, color_mask, 0.5, 0)

        # Encode mask to base64
        _, mask_buffer = cv2.imencode('.png', pred_mask)
        mask_base64 = base64.b64encode(mask_buffer).decode('utf-8')

        # Encode overlay to base64
        _, overlay_buffer = cv2.imencode('.png', overlay)
        overlay_base64 = base64.b64encode(overlay_buffer).decode('utf-8')

        return jsonify({
            'mask': mask_base64,
            'overlay': overlay_base64,
            'message': 'Mitochondria mask prediction successful'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Error processing image'
        }), 500