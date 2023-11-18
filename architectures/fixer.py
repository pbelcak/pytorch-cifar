import torch
from torch import nn
import math

class IFixable:
	def get_hardness(self) -> float:
		raise NotImplementedError()
	
	def set_hardness(self, hardness: float):
		raise NotImplementedError()

class FixerReLU(torch.nn.Module, IFixable):
	def __init__(self, init_hardness: float = 0.0):
		super().__init__()

		self.hardness = nn.Parameter(torch.empty(1,), requires_grad=False)
		self.hardness.data.fill_(init_hardness)

	def get_hardness(self) -> float:
		return self.hardness.data.item()

	def set_hardness(self, hardness: float):
		self.hardness.data.fill_(hardness)
	
	def forward(self, x):
		activated = torch.nn.functional.relu(x)

		out = (1-self.hardness) * activated + self.hardness * (x > 0.0).float()

		return out

class FixerNormedReLU(torch.nn.Module, IFixable):
	def __init__(self, init_hardness: float = 0.0):
		super().__init__()

		self.hardness = nn.Parameter(torch.empty(1,), requires_grad=False)
		self.hardness.data.fill_(init_hardness)
		self.mean = nn.Parameter(torch.empty(1,), requires_grad=True)
		self.mean.data.fill_(0.5)
		self.variance = nn.Parameter(torch.empty(1,), requires_grad=True)
		self.variance.data.fill_(0.5)

	def get_hardness(self) -> float:
		return self.hardness.data.item()

	def set_hardness(self, hardness: float):
		self.hardness.data.fill_(hardness)
	
	def forward(self, x):
		activated = torch.nn.functional.relu(x)
		activated_mean = activated.mean(dim=(-3,-2,-1)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
		activated_variance = activated.var(dim=(-3,-2,-1)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

		activated_normalized = (activated - activated_mean) / (activated_variance + 1e-12).sqrt()

		# out = ((x <= 0.0).float() + (1-self.hardness) * (x > 0.0).float()) * activated_normalized # original
		out = (1 - self.hardness * (x > 0.0).float()) * activated_normalized # should be the same as above but slightly faster
		out = out * self.variance.sqrt() + self.mean

		return out

class FixedModule(torch.nn.Module):
	def __init__(self, fixer: torch.nn.Module):
		super().__init__()
		self.fixer = fixer
	
	def fix(self, x):
		if self.fixer is not None:
			return self.fixer(x)
		else:
			return x
		
	def get_hardness(self) -> float:
		return self.fixer.get_hardness()
	
	def set_hardness(self, hardness: float):
		self.fixer.set_hardness(hardness)

	def add_hardness(self, hardness_increment: float):
		self.fixer.add_hardness(hardness_increment)

class Fixer(torch.nn.Module):
	def __init__(self, input_width: int, flare_coefficient: int):
		super().__init__()
		
		self.input_width = input_width
		init = 1.0 / math.sqrt(flare_coefficient)
		self.cuts = nn.Parameter(torch.empty(input_width, flare_coefficient).uniform_(-init, +init), requires_grad=True)

		self.hardness = nn.Parameter(torch.zeros(1), requires_grad=False)

		self.layer_norm = nn.LayerNorm(input_width * flare_coefficient, eps=1e-12)

	def get_hardness(self) -> float:
		return self.hardness.data.item()

	def set_hardness(self, hardness: float):
		self.hardness.data.fill_(hardness)
	
	def add_hardness(self, hardness_increment: float):
		self.hardness.data.add_(hardness_increment)

	def forward(self, x):
		# input has shape (..., input_width)
		x = x.unsqueeze(-1) # shape (..., input_width, 1)

		x = x.expand(-1, -1, self.cuts.shape[1]) # shape (..., input_width, flare_coefficient)
		cut_out = x - self.cuts
		activated_out = torch.nn.functional.relu(cut_out) # shape (..., input_width, flare_coefficient)

		out = (1-self.hardness) * activated_out + self.hardness * (cut_out > 0.0).float() # shape (..., input_width, flare_coefficient)

		return self.layer_norm(out.flatten(-2)) # shape (..., input_width * flare_coefficient)


class Prixer(nn.Module):
	pass