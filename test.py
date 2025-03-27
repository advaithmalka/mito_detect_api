import torch
import segmentation_models_pytorch as smp

model = smp.Unet(
        encoder_name="resnet50",        # choose encoder, e.g. mobilenet_v2 or resnet34
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=1,                  # model input channels (1 for grayscale)
        classes=1,                      # model output channels (1 for binary segmentation)
    )
model.load_state_dict(torch.load("./models/best_model_checkpoint.pt", map_location="cpu"))

# Apply dynamic quantization
model_quantized = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

torch.save(model_quantized.state_dict(), "./models/best_model_checkpoint_quantized.pt")