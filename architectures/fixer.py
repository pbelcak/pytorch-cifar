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
		activated = torch.clamp_max(activated, 1.0)

		interpolated = (1-self.hardness) * activated + self.hardness * (x > 0.0).float() * 0.5

		out = torch.where(torch.rand_like(x) < self.hardness.item(), interpolated, activated)

		return out
	
class FixerSigmoid(torch.nn.Module, IFixable):
	def __init__(self, init_hardness: float = 0.0):
		super().__init__()

		self.hardness = nn.Parameter(torch.empty(1,), requires_grad=False)
		self.hardness.data.fill_(init_hardness)

	def get_hardness(self) -> float:
		return self.hardness.data.item()

	def set_hardness(self, hardness: float):
		self.hardness.data.fill_(hardness)

	def forward(self, x):
		activated = torch.sigmoid(x)

		interpolated = (1-self.hardness) * activated + self.hardness * (x > 0.0).float()

		return interpolated

class ClampReLU(torch.nn.Module):
	def __init__(self):
		super().__init__()
	
	def forward(self, x):
		activated = torch.nn.functional.relu(x)

		out = torch.clamp_max(activated, 1.0)

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
		activated = torch.nn.functional.relu(x).clamp_max(1.0 / (self.hardness.data.item() + 0.1))
		hardened = (1 - self.hardness) * activated + self.hardness * (x > 0.0).float()
		hardened_mean = activated.mean(dim=(-3,-2,-1)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
		hardened_variance = activated.var(dim=(-3,-2,-1)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

		hardened_normalized = (hardened - hardened_mean) / (hardened_variance + 1e-12).sqrt()

		out = hardened_normalized * self.variance.sqrt() + self.mean

		return out
	
class LearnableQuantization(torch.nn.Module, IFixable):
	def __init__(self, n_cuts: int):
		super().__init__()
		self.cuts = nn.Linear(1, n_cuts, bias=True)
		self.vals = nn.Parameter(torch.empty(n_cuts,).uniform_(0, 1), requires_grad=True)
		self.hardness = nn.Parameter(torch.zeros(1), requires_grad=False)

		self.cuts.bias.data.uniform_(-0.5, +0.5)
		self.cuts.weight.data.uniform_(-1, +1)

	def get_hardness(self) -> float:
		return self.hardness.data.item()

	def set_hardness(self, hardness: float):
		self.hardness.data.fill_(hardness)

	def forward(self, x):
		x = x.unsqueeze(-1)

		cuts = self.cuts(x)
		activated = torch.sigmoid(cuts)
		interpolated = (1 - self.hardness) * activated + self.hardness * (cuts > 0.0).float()

		out = interpolated * self.vals
		out = out.mean(dim=-1)

		return out
	
class ReSigmoidLU(torch.nn.Module, IFixable):
	def __init__(self):
		super().__init__()
		self.hardness = nn.Parameter(torch.zeros(1), requires_grad=False)
	
	def get_hardness(self) -> float:
		return self.hardness.data.item()

	def set_hardness(self, hardness: float):
		self.hardness.data.fill_(hardness)

	def forward(self, x):
		activated = torch.clamp_max(torch.relu(x), 1.0)

		hardness = self.hardness.item() if self.training else 1.0

		interpolated = (1 - hardness) * activated + hardness * torch.sigmoid(x / (1.025 - hardness))

		return interpolated

	
class FixerMeticulouslyNormedReLU(torch.nn.Module, IFixable):
	def __init__(self, normalized_shape: tuple, init_hardness: float = 0.0):
		super().__init__()

		self.hardness = nn.Parameter(torch.empty(1,), requires_grad=False)
		self.hardness.data.fill_(init_hardness)
		self.targets = nn.Parameter(torch.empty(normalized_shape,).uniform_(0, 1), requires_grad=True)

	def get_hardness(self) -> float:
		return self.hardness.data.item()

	def set_hardness(self, hardness: float):
		self.hardness.data.fill_(hardness)
	
	def forward(self, x):
		activated = torch.nn.functional.relu(x)

		out = (1 - self.hardness) * activated + self.hardness * (x > 0.0).float() * self.targets

		return out
	
	def __repr__(self):
		return super().__repr__() + f" (hardness={self.get_hardness()}, shape={self.targets.shape})"
	
class FixerMagicReLU(torch.nn.Module, IFixable):
	def __init__(self, filters: int, kernel_size: tuple, init_hardness: float = 0.0):
		super().__init__()
		self.hardness = nn.Parameter(torch.empty(1,), requires_grad=False)
		self.hardness.data.fill_(init_hardness)
		self.targets = nn.Parameter(torch.empty(filters * kernel_size[0] * kernel_size[1],).uniform_(0, 1), requires_grad=True)
		self.unfold = nn.Unfold(kernel_size=kernel_size, dilation=(1,1), padding=(1,1), stride=(1,1))

	def get_hardness(self) -> float:
		return self.hardness.data.item()

	def set_hardness(self, hardness: float):
		self.hardness.data.fill_(hardness)

	def forward(self, x):
		# (batch_size, hiddem_dim, width, height)
		activated = torch.nn.functional.relu(x)

		# unfold
		windowed = self.unfold(activated) # (batch_size, hidden_dim * kernel_size[0] * kernel_size[1], blocks)

		# interpolate each block with the targets using self.hardness
		out = (1 - self.hardness) * windowed + self.hardness * self.targets.unsqueeze(0).unsqueeze(-1).expand(windowed.shape)

		# reshape
		out = out.reshape(activated.shape)

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