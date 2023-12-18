'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn

from .fixer import *
from .vnn import *

cfg = {
	'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
	'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
	'vgg19_q1_twopoint': [64, 64, 'M', 'Q1', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
	'vgg19_q1_full': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
	'vgg19_resilu_full': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
	'vgg19_resilu_block': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
	'vgg19_vnn': [64, 64, 'VNN', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module, IFixable):
	def __init__(self, vgg_name):
		super(VGG, self).__init__()
		self.features = self._make_layers(cfg[vgg_name])
		self.activation_class = ReSiLU if 'resilu' in vgg_name else ClampReLU
		self.final_fixer = Fixer(512, 1)
		self.classifier = nn.Linear(512, 10)
		self.hard_tensors = []

	def get_hardness(self) -> float:
		return self.final_fixer.get_hardness()
	
	def set_hardness(self, hardness: float):
		for feature in self.features:
			if isinstance(feature, IFixable):
				feature.set_hardness(hardness)
		self.final_fixer.set_hardness(hardness)

	def record_hard_tensor(self, x):
		self.hard_tensors += [ x.detach().cpu() ]

	def get_hard_tensors(self):
		return self.hard_tensors
	
	def clean_hard_tensors(self):
		self.hard_tensors = []

	def forward(self, x):
		for feature in self.features:
			x = feature(x)
			if isinstance(feature, ReSiLU):
				self.record_hard_tensor(x)
				if self.training:
					x = torch.where(torch.rand_like(x) < 0.1, (1-x).detach(), x)
			
		out = x
		out = out.view(out.size(0), -1)
		out = self.classifier(out)
		return out

	def _make_layers(self, cfg):
		layers = []
		in_channels = 3
		block = 0
		for x, nextx in zip(cfg, cfg[1:]+[None]):
			if x == 'M':
				layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
				block += 1
			elif x == 'VNN':
				block += 1
				layers += [
					nn.Flatten(),
					VNN(32 * 32 * 64, 16 * 16 * 64, 4, parallel_size=4),
					nn.Unflatten(1, (64, 16, 16)),
					nn.BatchNorm2d(min(64 * 2 ** (block-1), 512), track_running_stats=False),
				]
			elif x == 'Q1':
				layers += [
					LearnableQuantization(1),
					nn.BatchNorm2d(min(64 * 2 ** (block-1), 512), track_running_stats=False)
				]
			else:		
				if nextx == 'M':
					activation_class = ReSiLU
				else:
					activation_class = nn.ReLU
				
				layers += [
					nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
					activation_class(),
					nn.BatchNorm2d(x, track_running_stats=False),
				]
				in_channels = x
		layers += [
			# nn.AvgPool2d(kernel_size=1, stride=1),
			# activation_class()
		]
		return nn.ModuleList(layers)


def test():
	net = VGG('vgg19_vnn')
	x = torch.randn(2,3,32,32)
	y = net(x)
	print(y.size())

# test()
