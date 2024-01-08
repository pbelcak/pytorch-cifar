import torch
import torch.nn as nn

from .fixer import *

import math

import wandb


class Q1Linear(nn.Module, IFixable):
	def __init__(self, input_width: int, output_width: int, bias: bool = True, scale: bool = True):
		super(Q1Linear, self).__init__()
	
		self.input_width = input_width
		self.output_width = output_width

		init_k = 1.0 / math.sqrt(input_width)
		self.weight = nn.Parameter(torch.empty(input_width, output_width).uniform_(-init_k, +init_k), requires_grad=True)
		self.bias = nn.Parameter(torch.empty(output_width).uniform_(-init_k, +init_k), requires_grad=True) if bias else 0.0
		self.hardness = nn.Parameter(torch.zeros((1,)), requires_grad=False)
		self.scale = nn.Parameter(
			torch.empty(output_width).uniform_(-init_k, +init_k) if scale else torch.ones(output_width),
			requires_grad=scale
		)

	def get_hardness(self) -> float:
		return self.hardness.item()
	
	def set_hardness(self, hardness: float):
		self.hardness.fill_(hardness)

	def forward(self, x):
		hardness = self.hardness.item()

		if self.training:
			interpolated = (1 - hardness) * self.weight + hardness * torch.sigmoid(self.weight * 100 / (1.025 - hardness))
		else:
			interpolated = (self.weight > 0).float()

		#wandb.log({
		#	'mean_weight_distance': torch.abs(interpolated-(self.weight > 0).float()).mean().item(),
		#})

		return torch.mm(x, interpolated) * self.scale + self.bias

class S1Linear(nn.Module, IFixable):
	def __init__(self, input_width: int, output_width: int, bias: bool = True, scale: bool = True):
		super(S1Linear, self).__init__()
	
		self.input_width = input_width
		self.output_width = output_width

		init_k = 1.0 / math.sqrt(input_width)
		self.init_k = init_k
		self.weight = nn.Parameter(torch.empty(input_width, output_width).uniform_(-init_k, +init_k), requires_grad=True)
		self.bias = nn.Parameter(torch.empty(output_width).uniform_(-init_k, +init_k), requires_grad=True) if bias else 0.0
		self.hardness = nn.Parameter(torch.zeros((1,)), requires_grad=False)
		self.scale = nn.Parameter(
			torch.empty(output_width).uniform_(-init_k, +init_k) if scale else torch.ones(output_width),
			requires_grad=scale
		)

	def get_hardness(self) -> float:
		return self.hardness.item()
	
	def set_hardness(self, hardness: float):
		self.hardness.fill_(hardness)

	def forward(self, x):
		hardness = self.hardness.item()

		if self.training:
			interpolated = (1 - hardness) * self.weight + hardness * (torch.sigmoid(self.weight * 100 / (1.025 - hardness)) * 2 - 1)
		else:
			interpolated = (self.weight > 0).float() * 2 - 1

		return torch.mm(x, interpolated) * self.scale + self.bias

# test()

class Q1FF(nn.Module, IFixable):
	def __init__(self, input_width: int, output_width: int):
		super().__init__()

		self.layers = nn.ModuleList([
			nn.Flatten(),
			nn.Linear(input_width, 8192, True),
			nn.LayerNorm((8192,), eps=1e-12),
			nn.ReLU(),
			Q1Linear(8192, 8192, True),
			nn.LayerNorm((8192,), eps=1e-12),
			nn.ReLU(),
			nn.Linear(8192, output_width, True),
		])

	def get_hardness(self) -> float:
		return self.layers[4].get_hardness()
	
	def set_hardness(self, hardness: float):
		for l in self.layers:
			if isinstance(l, IFixable):
				l.set_hardness(0.9 + 0.1 * hardness)

	def forward(self, x):
		for l in self.layers:
			x = l(x)

		return x

class ResiQ1FF(nn.Module, IFixable):
	def __init__(self, input_width: int, output_width: int):
		super().__init__()

		self.flatten = nn.Flatten()
		self.linear1 = S1Linear(input_width, 8192, True)
		self.act = nn.ReLU()
		self.ln1 = nn.LayerNorm((8192,), eps=1e-12)

		self.linear2 = Q1Linear(8192, 8192, True)
		self.ln2 = nn.LayerNorm((8192,), eps=1e-12)

		self.linear3 = Q1Linear(8192, output_width, True)

	def get_hardness(self) -> float:
		return self.linear2.get_hardness()
	
	def set_hardness(self, hardness: float):
		hardness = 0.9 + 0.1 * hardness
		self.linear1.set_hardness(hardness)
		self.linear2.set_hardness(hardness)
		self.linear3.set_hardness(hardness)

	def forward(self, x):
		x = self.flatten(x)

		y = self.linear1(x)
		y = self.act(y)
		x = self.ln1(y)

		y = self.linear2(x)
		y = self.act(y)
		x = self.ln2(y + x)

		x = self.linear3(x)
		
		return x


	
class S1FF(nn.Module, IFixable):
	def __init__(self, input_width: int, output_width: int):
		super().__init__()

		self.layers = nn.ModuleList([
			nn.Flatten(),
			nn.Linear(input_width, 8192, True),
			nn.LayerNorm((8192,), eps=1e-12),
			nn.ReLU(),
			S1Linear(8192, 8192, True),
			nn.LayerNorm((8192,), eps=1e-12),
			nn.ReLU(),
			S1Linear(8192, output_width, True),
			nn.LayerNorm((output_width,), eps=1e-12)
		])

	def get_hardness(self) -> float:
		return self.layers[4].get_hardness()
	
	def set_hardness(self, hardness: float):
		for l in self.layers:
			if isinstance(l, IFixable):
				l.set_hardness(0.9 + 0.1 * hardness)

	def forward(self, x):
		for l in self.layers:
			x = l(x)

		return x
	
class ClassicFF(nn.Module, IFixable):
	def __init__(self, input_width: int, output_width: int):
		super().__init__()

		self.layers = nn.ModuleList([
			nn.Flatten(),
			nn.Linear(input_width, 4096, bias=True),
			nn.ReLU(),
			nn.Linear(4096, 4096, bias=True),
			nn.ReLU(),
			nn.Linear(4096, output_width, bias=True)
		])

	def get_hardness(self) -> float:
		return 0.0
	
	def set_hardness(self, hardness: float):
		pass

	def forward(self, x):
		for l in self.layers:
			x = l(x)

		return x