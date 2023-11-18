'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn

from .fixer import *

cfg = {
	'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
	'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
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
		past_first_block = False
		for x in cfg:
			if x == 'M':
				layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
				past_first_block = True
			else:
				layers += [
					nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
					nn.BatchNorm2d(x),
					(FixerNormedReLU() if past_first_block else nn.ReLU(inplace=True)),
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
