'''FF for Vision; in Pytorch.'''
import torch
import torch.nn as nn

from .fixer import *

cfg = {
	'ff1024': [256, 'RS', 64, 'RS'],
}


class FF(nn.Module, IFixable):
	def __init__(self, input_width: int, output_width: int, name: str, flare: int = 1):
		super(FF, self).__init__()
	
		self.input_width = input_width
		self.output_width = output_width

		chosen_cfg = cfg[name]
		self.features = self._make_layers(chosen_cfg)
		self.final_fixer = Fixer(chosen_cfg[-2], flare)
		self.classifier = nn.Linear(chosen_cfg[-2] * flare, self.output_width)

	def get_hardness(self) -> float:
		return self.final_fixer.get_hardness()
	
	def set_hardness(self, hardness: float):
		for feature in self.features:
			if isinstance(feature, IFixable):
				feature.set_hardness(hardness)
		self.final_fixer.set_hardness(hardness)

	def forward(self, x):
		x = x.flatten(1)
		out = self.features(x)
		out = out.view(out.size(0), -1)
		# out = self.final_fixer(out)
		out = self.classifier(out)
		return out

	def _make_layers(self, cfg):
		layers = []
		in_width = self.input_width
		block = 0
		for x in cfg:
			if x == 'R':
				layers += [ ClampReLU(), ]
				block += 1
			elif x == 'S':
				layers += [ torch.nn.Sigmoid(), ]
				block += 1
			elif x == 'RS':
				layers += [ ReSigmoidLU(), nn.LayerNorm((in_width,), eps=1e-12) ]
				block += 1
			elif x == 'Q1':
				layers += [
					LearnableQuantization(1),
					nn.LayerNorm((in_width,), eps=1e-12)
				]
			else:
				layers += [
					nn.Linear(in_width, x)
				]
				in_width = x
		
		return nn.Sequential(*layers)


def test():
	net = FF(784, 10, 'ff1024')
	x = torch.randn(4,28,28)
	y = net(x)
	print(y.size())

# test()
