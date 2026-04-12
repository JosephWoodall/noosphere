import torch
import numpy as np
from s4_eeg import S4EEGEncoder

def test_encoder_flow():
    print("Initializing Riemannian-S4 Encoder...")
    # 21 channels, 256 samples (1 second)
    encoder = S4EEGEncoder(in_channels=21, d_model=128, n_actions=4)
    encoder.eval()
    
    # Create dummy EEG batch: (Batch, Channels, Time)
    dummy_input = torch.randn(2, 21, 256)
    
    print(f"Input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        out = encoder(dummy_input)
        
    print("\nOutput Check:")
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")
        else:
            print(f"  {k}: {type(v)}")
            
    # Verify critical components
    assert "intent_probs" in out, "Missing intent_probs"
    assert out["intent_probs"].shape == (2, 4), "Wrong output probability shape"
    print("\nSUCCESS: Tensor flow verified (Riemannian + Temporal fusion works).")

if __name__ == "__main__":
    test_encoder_flow()
