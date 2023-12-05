from greedy import DenseSumPredictor
from torch import nn
import torch

from .fixer import *

class DenseSumUnit(nn.Module, IFixable):
	def __init__(self, input_width: int):
		super().__init__()

		self.predictor = DenseSumPredictor(input_width)
		# self.activation = ReSiLU()

	def get_hardness(self) -> float:
		return self.predictor.get_hardness()
	
	def set_hardness(self, hardness: float):
		self.predictor.set_hardness(hardness)

	def forward(self, input: torch.Tensor):
		# input shape is (batch_size, input_width)

		# predict the output
		output = self.predictor(input)

		# apply the activation function
		# output = self.activation(output)

		return output