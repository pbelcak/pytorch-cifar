import torch
import torch.nn as nn

from .fixer import *

class PopcntLayer(nn.Module, IFixable):
	def __init__(self, input_width: int, output_width: int, popcnt_width: int=64):
		super(PopcntLayer, self).__init__()
	
		self.input_width = input_width
		self.popcnt_width = popcnt_width
		self.output_width = output_width
		self.input_selection = nn.Parameter(torch.randint(0, input_width, (output_width, popcnt_width)), requires_grad=False)	
		self.weights = nn.Parameter(torch.randn(output_width, popcnt_width), requires_grad=True)
		self.weight_activation = ReSiLU()
		self.biases = nn.Parameter(torch.randn(output_width) * popcnt_width/3 + popcnt_width/4, requires_grad=True)

		self.activation = ReSiLU()

	def get_hardness(self) -> float:
		return self.weight_activation.get_hardness()
	
	def set_hardness(self, hardness: float):
		self.weight_activation.set_hardness(hardness)
		self.activation.set_hardness(hardness)

	def forward(self, x):
		selected = x[:, self.input_selection.flatten()]
		weighted = selected * self.weight_activation(self.weights.flatten()).unsqueeze(0)
		weighted = weighted.reshape(weighted.shape[0], self.output_width, self.popcnt_width)
		summed = weighted.sum(dim=-1)
		out = self.activation(summed - self.biases.flatten())

		return out

class Popcnt(nn.Module, IFixable):
	def __init__(self, input_width: int, output_width: int):
		super(Popcnt, self).__init__()
	
		self.input_width = input_width
		self.output_width = output_width

		self.layers = nn.ModuleList([
			PopcntLayer(128*5*5, 128*64, 128),
			nn.LayerNorm((128*64,), eps=1e-12),
			PopcntLayer(128*64, 128*64, 128),
			nn.LayerNorm((128*64,), eps=1e-12),
			PopcntLayer(128*64, 64*64, 128),
		])

	def get_hardness(self) -> float:
		return self.layers[0].get_hardness()
	
	def set_hardness(self, hardness: float):
		for layer in self.layers:
			if isinstance(layer, IFixable):
				layer.set_hardness(hardness)

	def forward(self, x):
		for layer in self.layers:
			x = layer(x)

		out = x.view(x.size(0), 4*64, -1).sum(dim=-1) - 8
		return out

def test():
	net = Popcnt(784, 10, 'ff1024')
	x = torch.randn(4,28,28)
	y = net(x)
	print(y.size())

# test()
