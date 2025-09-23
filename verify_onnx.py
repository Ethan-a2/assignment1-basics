import torch
import onnxruntime as ort
import numpy as np

def verify_onnx_model():
    # Load the ONNX model
    onnx_model_path = "tests/fixtures/ts_tests/model.onnx"
    session = ort.InferenceSession(onnx_model_path)
    
    # Create test input
    batch_size = 1
    seq_length = 16
    vocab_size = 10000
    
    # Create random input indices
    input_indices = np.random.randint(0, vocab_size, (batch_size, seq_length)).astype(np.int64)
    
    # Run inference with ONNX model
    onnx_outputs = session.run(None, {'input_indices': input_indices})
    
    # Load original PyTorch model for comparison
    import sys
    import json
    from pathlib import Path
    sys.path.append(str(Path(__file__)))
    
    from cs336_basics.transformerLM import TransformerLM
    
    # Load model configuration
    with open("tests/fixtures/ts_tests/model_config.json", 'r') as f:
        config = json.load(f)
    
    # Create model instance
    model = TransformerLM(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        context_length=config["context_length"],
        theta=config["rope_theta"],
        num_layers=config["num_layers"]
    )
    
    # Load state dict
    state_dict = torch.load("tests/fixtures/ts_tests/model.pt", map_location="cpu")
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    
    # Load state dict into model
    model.load_state_dict(state_dict)
    model.eval()
    
    # Run inference with PyTorch model
    with torch.no_grad():
        torch_input = torch.from_numpy(input_indices)
        torch_outputs = model(torch_input)
    
    # Compare outputs
    onnx_logits = onnx_outputs[0]
    torch_logits = torch_outputs.numpy()
    
    print(f"ONNX output shape: {onnx_logits.shape}")
    print(f"PyTorch output shape: {torch_logits.shape}")
    
    # Check if outputs are close
    diff = np.abs(onnx_logits - torch_logits).max()
    print(f"Max difference between ONNX and PyTorch outputs: {diff}")
    
    if diff < 1e-4:
        print("SUCCESS: ONNX model output matches PyTorch model output!")
    else:
        print("WARNING: ONNX model output differs from PyTorch model output.")
    
    return onnx_logits

if __name__ == "__main__":
    verify_onnx_model()