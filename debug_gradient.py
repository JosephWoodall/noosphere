import torch
from noosphere.s4_eeg import S4EEGEncoder, DirichletEDLLoss
import torch.nn.functional as F

torch.manual_seed(42)

model = S4EEGEncoder(in_channels=21, d_model=192, n_blocks=3, d_state=64, n_actions=4).cuda()
model.train()

x = torch.randn(8, 21, 256, device='cuda')
y = torch.randint(0, 4, (8,), device='cuda')
y_ohe = F.one_hot(y, num_classes=4).float()

with torch.amp.autocast('cuda'):
    out = model(x)
    alpha = out["alpha"].float()
    
loss_fn = DirichletEDLLoss(n_classes=4, annealing_step=30).cuda()
loss = loss_fn(alpha, y_ohe, 1)

print(f"Forward completed. Loss: {loss.item()}")

loss.backward()

nan_grads = {}
for name, p in model.named_parameters():
    if p.grad is not None and torch.isnan(p.grad).any():
        nan_grads[name] = True
    elif p.grad is None:
        nan_grads[name] = "None"
        
if nan_grads:
    print("Detected NaN gradients in parameters:")
    for k, v in nan_grads.items():
        if v is True:
            print(f"- {k} (NaN)")
else:
    print("All gradients are finite and valid.")
    
# Let's also print statistics of output
print(f"Intent Probs: {out['intent_probs']}")
print(f"Alpha: {alpha}")
