import torch
import torch.nn as nn

class LogisticRegression(nn.Module):
  def __init__(self):
    super(LogisticRegression, self).__init__()
    self.linear = torch.nn.Linear(165, 3)
    self.act = torch.nn.Sigmoid()
  def forward(self, x):
    o = self.linear(x)
    o = self.act(o)
    return o