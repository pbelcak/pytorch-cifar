from .fixer import *

import torch
from torch import nn

import math


class LLUT(nn.Module, IFixable):
	def __init__(self, input_width, output_width, n_cuts):
		super().__init__()

		self.input_width = input_width
		self.output_width = output_width
		self.n_cuts = n_cuts

		init_k = 1.0 / math.sqrt(input_width)
		self.cuts = nn.Parameter(torch.empty(n_cuts, input_width).uniform_(-init_k, +init_k), requires_grad=True)
		self.cut_weights = nn.Parameter(torch.empty(n_cuts, 2, output_width,).uniform_(-init_k, +init_k), requires_grad=True)
		self.outputs = nn.Parameter(torch.empty(2 ** n_cuts, self.output_width).uniform_(-init_k, +init_k), requires_grad=True)

		self.cut_activation = ReSiLU()

	def forward(self, x):
		# x has shape (batch_size, input_width)
		batch_size = x.shape[0]

		# compute the cut logits
		cut_logits = torch.mm(x, self.cuts.transpose(0, 1)) # shape (batch_size, n_cuts,)
		cut_activations = torch.clamp(self.cut_activation(cut_logits), 0.0, 1.0) # shape (batch_size, n_cuts,)
		cut_counteractivations = 1 - cut_activations

		cut_output_scalers = torch.stack([cut_activations, cut_counteractivations], dim=-1) # shape (batch_size, n_cuts, 2)
		cut_output_contributions = (self.cut_weights.unsqueeze(0) * cut_output_scalers.unsqueeze(-1)).sum(dim=2).sum(dim=1) # shape (batch_size, output_width)

		current_output_indices = torch.zeros((batch_size,), dtype=torch.long, device=x.device)
		cut_activations = torch.round(cut_activations).long()

		for depth in range(0, self.n_cuts):
			current_output_indices = 2 * current_output_indices + cut_activations[:, depth]

		outputs = self.outputs.index_select(0, index=current_output_indices) # shape (batch_size, output_width)

		# TODO: remember to turn this into += if uncommenting the above
		outputs += cut_output_contributions

		return outputs

	@torch.no_grad()
	def set_hardness(self, hardness: float):
		self.cut_activation.set_hardness(hardness)

	def get_hardness(self) -> float:
		return self.hardness.data.item()

class MultiLLUT(nn.Module):
	def __init__(self, input_width, output_width, n_cuts, multiplicity: int = 1):
		super().__init__()

		self.multiplicity = multiplicity

		self.lluts = nn.ModuleList([
			LLUT(input_width, output_width, n_cuts) for _ in range(multiplicity)
		])

	def forward(self, x):
		out = self.lluts[0](x)

		for i in range(1, self.multiplicity):
			out += self.lluts[i](x)

		return out / self.multiplicity