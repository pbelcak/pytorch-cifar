'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn

from .fixer import *

cfg = {
	'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
	'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
	'vgg19_q1_twopoint': [64, 64, 'M', 'Q1', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
	'vgg19_q1_full': [64, 64, 'M', 'Q1', 128, 128, 'M', 'Q1', 256, 256, 256, 256, 'M', 'Q1', 512, 512, 512, 512, 'M', 'Q1', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module, IFixable):
	def __init__(self, vgg_name, flare: int = 1):
		super(VGG, self).__init__()
		self.features = self._make_layers(cfg[vgg_name])
		self.final_fixer = Fixer(512, flare)
		self.classifier = nn.Linear(512 * flare, 10)

	def get_hardness(self) -> float:
		return self.final_fixer.get_hardness()
	
	def set_hardness(self, hardness: float):
		for feature in self.features:
			if isinstance(feature, IFixable):
				feature.set_hardness(hardness)
		self.final_fixer.set_hardness(hardness)

	def forward(self, x):
		out = self.features(x)
		out = out.view(out.size(0), -1)
		out = self.final_fixer(out)
		out = self.classifier(out)
		return out

	def _make_layers(self, cfg):
		layers = []
		in_channels = 3
		block = 0
		for x in cfg:
			if x == 'M':
				layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
				block += 1
			elif x == 'Q1':
				layers += [
					LearnableQuantization(1),
					nn.BatchNorm2d(min(64 * 2 ** (block-1), 512), track_running_stats=False)
				]
			else:
				# past_first_block = block > 0
				# we need the shape for normalization
				# we start at (b_s, 32, 32); then halve for every increment of block
				shape = (
					min(64 * 2 ** block, 512),
					32 // (2 ** block),
					32 // (2 ** block)
				)
				
				layers += [
					nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
					nn.BatchNorm2d(x, track_running_stats=False),
					ClampReLU(),
				]
				in_channels = x
		layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
		return nn.Sequential(*layers)


def test():
	net = VGG('VGG11')
	x = torch.randn(2,3,32,32)
	y = net(x)
	print(y.size())

# test()
