import numpy as np
import torch


def test_onnx_model_inference(ts_onnx_model, ts_state_dict):
    """Test that demonstrates how to use the ONNX model for inference."""
    _, config = ts_state_dict
    
    # Create test input
    batch_size = 1
    seq_length = config["context_length"]
    vocab_size = config["vocab_size"]
    
    # Create random input indices
    input_indices = np.random.randint(0, vocab_size, (batch_size, seq_length)).astype(np.int64)
    
    # Run inference with ONNX model
    onnx_outputs = ts_onnx_model.run(None, {'input_indices': input_indices})
    
    # Check output shape
    assert len(onnx_outputs) == 1
    logits = onnx_outputs[0]
    assert logits.shape == (batch_size, seq_length, vocab_size)
    
    print(f"ONNX model inference successful!")
    print(f"Input shape: {input_indices.shape}")
    print(f"Output shape: {logits.shape}")