import torch
import torch.nn as nn

import math

from .fixer import *

class VNN(nn.Module, IFixable):
	def __init__(self, input_width, output_width, depth, parallel_size, activation=nn.GELU):
		super().__init__()

		self.input_width = input_width
		self.output_width = output_width
		self.depth = depth
		self.parallel_size = parallel_size
		self.n_nodes = 2 ** (self.depth + 1) - 1

		self.linear_in = nn.Linear(input_width, parallel_size * self.n_nodes, bias=True)
		self.linear_out = nn.Linear(parallel_size * self.n_nodes, output_width, bias=False)
		self.hardness = nn.Parameter(torch.zeros((1,)), requires_grad=False)
		self.last_hardness = nn.Parameter(torch.zeros((1,)), requires_grad=False)

		self.activation = activation()

	def forward(self, oldx: torch.Tensor) -> torch.Tensor:
		# x has shape (..., input_width)
		x = oldx.reshape(-1, self.input_width)
		# x has shape (batch_size, input_width)
		batch_size = x.shape[0]
		
		logits = self.linear_in(x)
		logit_decisions = (logits > 0).long() # (batch_size, parallel_size * n_nodes)

		# recursively descending by depth, enforce conditionality
		decisions = logit_decisions.view(batch_size, self.parallel_size, self.n_nodes) # (batch_size, parallel_size, n_nodes)
		with torch.no_grad():
			current_nodes = torch.zeros((batch_size, self.parallel_size), dtype=torch.long, device=x.device)
			decision_map = torch.zeros_like(decisions, dtype=torch.float) # (batch_size, parallel_size, n_nodes)
			decision_map.scatter_(dim=2, index=current_nodes.unsqueeze(-1), value=1.0)

			for d in range(self.depth):
				current_platform = 2 ** d - 1
				next_platform = 2 ** (d + 1) - 1
				moves = torch.gather(decisions, 2, current_nodes.unsqueeze(2)).squeeze(2)
				next_nodes = (current_nodes - current_platform) * 2 + moves + next_platform
				decision_map.scatter_(2, next_nodes.unsqueeze(-1), 1.0)
				current_nodes = next_nodes
				
		activations = self.activation(logits) # (batch_size, parallel_size * n_nodes)

		decision_map_flat = decision_map.flatten(1, 2) # (batch_size, parallel_size * n_nodes)
		new_logits = self.linear_out(
			((1.0 - self.hardness) * activations + self.hardness * logit_decisions.float()) * decision_map_flat
		) # (batch_size, parallel_size * n_nodes)

		ret = new_logits.reshape_as(oldx)
		return ret

	@torch.no_grad()
	def set_hardness(self, hardness: float):
		self.hardness.data = torch.tensor(hardness, device=self.hardness.device)

    def get_hardness(self) -> float:
		return self.hardness.data.item()
