import torch
import torch.nn as nn
import torch.nn.functional as F



input = torch.randn(3, requires_grad=True)
target = torch.empty(3).random_(2)
loss = F.binary_cross_entropy_with_logits(input, target)
loss.backward()
print (loss)