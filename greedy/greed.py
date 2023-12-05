import torch
from torch import nn

import math

class Greed(nn.Module):
	def __init__(self, predictors: list = None):
		super().__init__()

		self.predictors = nn.ModuleList(predictors) if predictors is not None else nn.ModuleList([])

	def forward(self, input: torch.Tensor):
		# input shape is (batch_size, input_width)

		# iteratively predict the output
		current_nodes = torch.zeros((input.shape[0],), dtype=torch.long, device=input.device)
		depth = math.floor(math.log2(len(self.predictors)))

		for d in range(depth):
			current_predictors = self.predictors[current_nodes.tolist()]
			pass
			# TODO: return here once training code is done

	