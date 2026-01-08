import torch
import torch.onnx
from test_xception import load_patched_model

# Charger le modele PyTorch
MODEL_PATH = './model_github/deepfake-detection-with-xception/models/x-model23.p'
model = load_patched_model(MODEL_PATH)

if model:
    # 299x299
    dummy_input = torch.randn(1, 3, 299, 299)

    # ONNX
    onnx_path = "xception_faceforensics.onnx"
    print(f"Converting to {onnx_path}...")
    
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        verbose=True,
        input_names=['input_image'], 
        output_names=['output_class'],
        opset_version=11 # compatible TF
    )
    print("done")