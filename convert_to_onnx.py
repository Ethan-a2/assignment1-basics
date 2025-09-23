import torch
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from cs336_basics.transformerLM import TransformerLM

def convert_model_to_onnx():
    # Define paths
    fixtures_path = Path(__file__).parent / "tests" / "fixtures" / "ts_tests"
    model_pt_path = fixtures_path / "model.pt"
    model_config_path = fixtures_path / "model_config.json"
    model_onnx_path = fixtures_path / "model.onnx"
    
    # Load model configuration
    with open(model_config_path, 'r') as f:
        config = json.load(f)
    
    print("Model configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
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
    state_dict = torch.load(model_pt_path, map_location="cpu")
    # Remove the _orig_mod. prefix if present
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    
    # Load state dict into model
    model.load_state_dict(state_dict)
    model.eval()
    
    # Create dummy input for ONNX export
    batch_size = 1
    seq_length = config["context_length"]
    dummy_input = torch.randint(0, config["vocab_size"], (batch_size, seq_length))
    
    print(f"Dummy input shape: {dummy_input.shape}")
    
    # Export to ONNX
    print("Exporting to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        model_onnx_path,
        export_params=True,
        opset_version=14,  # Updated to version 14 for aten::triu support
        do_constant_folding=True,
        input_names=['input_indices'],
        output_names=['logits'],
        dynamic_axes={
            'input_indices': {0: 'batch_size', 1: 'sequence_length'},
            'logits': {0: 'batch_size', 1: 'sequence_length'}
        }
    )
    
    print(f"Model successfully converted to ONNX format: {model_onnx_path}")

if __name__ == "__main__":
    convert_model_to_onnx()