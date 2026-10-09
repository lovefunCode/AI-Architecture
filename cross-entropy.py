import torch
import torch.nn.functional as F

logits = torch.tensor([2.0, 1.0, 0.1])

target = torch.tensor(0)

loss = F.cross_entropy(
    logits.unsqueeze(0),
    target.unsqueeze(0)
)

print('loss-', loss)

# python3 cross-entropy.py