import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import List, Dict
import time

from noosphere.s4_eeg import S4EEGEncoder, DirichletEDLLoss
from noosphere.synth import ScalpEEGGenerator

def run_bci_performance_benchmark(n_train: int = 5000, n_test: int = 500, epochs: int = 100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting BCI Performance Benchmark on {device}...")
    
    n_actions = 4 # REST, RIGHT, LEFT, BOTH
    encoder = S4EEGEncoder(in_channels=3, d_model=256, n_blocks=4, n_actions=n_actions).to(device)
    
    # We use a custom mapping for synthetic intents to our 4 classes
    intent_map = {
        ScalpEEGGenerator.INTENT_REST: 0,
        ScalpEEGGenerator.INTENT_RIGHT_HAND: 1,
        ScalpEEGGenerator.INTENT_LEFT_HAND: 2,
        ScalpEEGGenerator.INTENT_BOTH_HANDS: 3
    }
    
    def get_data(n: int):
        gen = ScalpEEGGenerator(seed=np.random.randint(1000))
        X, Y = [], []
        intents = list(intent_map.keys())
        for _ in range(n):
            intent = np.random.choice(intents)
            seg = gen.next_segment(intent=intent)
            X.append(seg["eeg"])
            Y.append(intent_map[intent])
        return torch.tensor(np.array(X), device=device), torch.tensor(np.array(Y), device=device)

    from torch.utils.data import DataLoader, TensorDataset
    
    X_train, Y_train = get_data(n_train)
    X_test, Y_test = get_data(n_test)
    
    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=16, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=16)
    
    optimizer = optim.AdamW(encoder.parameters(), lr=1e-3)
    # criterion = DirichletEDLLoss(n_classes=n_actions)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Training on {n_train} segments...")
    for epoch in range(epochs):
        encoder.train()
        epoch_loss = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            out = encoder(x_batch)
            # alpha = out["alpha"]
            # y_one_hot = F.one_hot(y_batch, num_classes=n_actions).float()
            # loss = criterion(alpha, y_one_hot, epoch)
            logits = out["intent_probs"] # out["intent_probs"] are already probs, but CE expects logits if using CrossEntropyLoss
            # Actually, S4EEGEncoder.forward returns probs. I should use NLLLoss or change it to return logits.
            
            # Let's use NLLLoss and log_softmax if it returns probs
            # Or just use the evidence/alpha for something else.
            
            # Let's fix S4EEGEncoder to return logits too, or just use the intent_probs with CrossEntropy
            # Actually CrossEntropyLoss(probs) is not standard.
            
            # I'll use the intent_proj output directly if I can.
            loss = criterion(torch.log(out["intent_probs"] + 1e-8), y_batch)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Avg Loss: {epoch_loss/len(train_loader):.4f}")

    print("\nEvaluating...")
    encoder.eval()
    all_preds = []
    all_targets = []
    all_conf = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            out = encoder(x_batch)
            all_preds.append(out["intent_probs"].argmax(dim=-1))
            all_targets.append(y_batch)
            all_conf.append(out["confidence"])
            
        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)
        acc = (preds == targets).float().mean().item()
        conf = torch.cat(all_conf).mean().item()
        
        print(f"Test Accuracy: {acc*100:.2f}%")
        print(f"Mean Confidence: {conf:.4f}")
        
        # Per-class accuracy
        for i in range(n_actions):
            mask = (Y_test == i)
            if mask.sum() > 0:
                class_acc = (preds[mask] == Y_test[mask]).float().mean().item()
                print(f"  Class {i} Accuracy: {class_acc*100:.2f}%")

    if acc >= 0.85:
        print("\nPASSED: Performance target of 85% achieved on synthetic data.")
    else:
        print(f"\nFAILED: Performance target of 85% not met ({acc*100:.2f}%).")
    
    return acc

if __name__ == "__main__":
    run_bci_performance_benchmark()
