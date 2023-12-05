import torch
from torch import nn

from .predictor import Predictor

class DenseSumPredictor(Predictor):
	def __init__(self, input_width: int):
		super().__init__()

		self.input_width = input_width

		self.choices = nn.Parameter(torch.empty(input_width, 4).uniform_(-1, +1), requires_grad=True)
		self.bias = nn.Parameter(torch.empty(1).uniform_(0, self.input_width), requires_grad=True)
		self.hardness = nn.Parameter(torch.zeros(1), requires_grad=False)

	def get_hardness(self) -> float:
		return self.hardness.data.item()
	def set_hardness(self, hardness: float):
		self.hardness.data.fill_(hardness)

	def forward(self, input: torch.Tensor):
		# input shape is (batch_size, input_width)

		constant_1 = torch.ones_like(input)
		constant_0 = torch.zeros_like(input)
		identity = input
		inverse = 1 - input

		# mix constant_1, constant_0, identity, and inverse according to the softmax of self.choices
		input_variants = torch.stack([constant_1, constant_0, identity, inverse], dim=-1) # shape is (batch_size, input_width, 4)
		
		maxima = torch.argmax(self.choices, dim=-1) # shape is (input_width,)
		hard_mixtures = torch.nn.functional.one_hot(maxima, num_classes=4).float() # shape is (input_width, 4)

		if self.training:
			mixtures = torch.softmax(self.choices, dim=-1) # shape is (input_width, 4)
			interpolated_mixtures = (1 - self.hardness) * mixtures + self.hardness * hard_mixtures
			mixtures = interpolated_mixtures
		else:
			mixtures = hard_mixtures
		
		mixed_input_variants = (input_variants * mixtures.unsqueeze(0)).sum(dim=-1) # shape is (batch_size, input_width, 4) -> (batch_size, input_width)

		# sum the mixed input variants
		output = mixed_input_variants.sum(dim=-1) # shape is (batch_size, input_width) -> (batch_size)
		biased = output - self.bias

		return biased
		