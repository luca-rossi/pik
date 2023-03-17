import torch.nn as nn

class LinearProbe(nn.Module):
	def __init__(self, dims):
		super().__init__()
		self.model = nn.Sequential(
			nn.Linear(dims, 1),
			nn.Sigmoid(),
		)

	def forward(self, x):
		return self.model(x)
