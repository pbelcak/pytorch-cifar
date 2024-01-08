'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn

from .fixer import *
from .vnn import *
from .llut import *

cfg = {
	'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
	'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
	'vgg19_q1_twopoint': [64, 64, 'M', 'Q1', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
	'vgg19_q1_full': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
	'vgg19_resilu_full': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
	'vgg19_resilu_block': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
	'vgg19_vnn': [64, 64, 'VNN', 128, 128, 'MP', 256, 256, 256, 256, 'MP', 512, 512, 512, 512, 'MP', 512, 512, 512, 512, 'MP'],
	'vgg19_llut': [64, 64, 'LLUT', 128, 128, 'MP', 256, 256, 256, 256, 'MP', 512, 512, 512, 512, 'MP', 512, 512, 512, 512, 'MP'],
	'vgg_llut_fin': [64, 64, 'LLUTfin', ],
	'vgg_vnn_fin': [64, 64, 'VNNfin', ]
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
			#if isinstance(feature, ReSiLU):
				# self.record_hard_tensor(x)
			#	if self.training:
			#		x = torch.where(torch.rand_like(x) < 0.1, (1-x).detach(), x)
			
		out = x
		out = out.view(out.size(0), -1)
		out = self.classifier(out)
		return out

	def _make_layers(self, cfg):
		layers = []
		in_channels = 3
		block = 0
		for x, nextx in zip(cfg, cfg[1:]+[None]):
			if x == 'M' or x == 'MP':
				layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
				block += 1
			elif x == 'VNN':
				block += 1
				layers += [
					nn.AvgPool2d(kernel_size=2, stride=2),
					nn.Flatten(),
					VNN(16 * 16 * 64, 16 * 16 * 64, 9, parallel_size=4),
					nn.Unflatten(1, (64, 16, 16)),
					nn.BatchNorm2d(min(64 * 2 ** (block-1), 512), track_running_stats=False),
				]
			elif x == 'VNNfin':
				block += 1
				layers += [
					nn.AvgPool2d(kernel_size=2, stride=2),
					nn.Flatten(),
					VNN(16 * 16 * 64, 512, 7, parallel_size=4),
					nn.LayerNorm((512,), eps=1e-12)
				]
			elif x == 'LLUT':
				block += 1
				layers += [
					nn.AvgPool2d(kernel_size=2, stride=2),
					nn.Flatten(),
					MultiLLUT(16 * 16 * 64, 16 * 16 * 64, 8, 4),
					nn.Unflatten(1, (64, 16, 16)),
					nn.BatchNorm2d(min(64 * 2 ** (block-1), 512), track_running_stats=False),
				]
			elif x == 'LLUTfin':
				block += 1
				layers += [
					nn.AvgPool2d(kernel_size=2, stride=2),
					nn.Flatten(),
					MultiLLUT(16 * 16 * 64, 512, 8, 4),
					nn.LayerNorm((512,), eps=1e-12)
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

ae_cfgs = {
	'ae_sanity32': [ 32, 32, 3],
	'ae_sanity128': [ 128, 128, 3],
	'ae_4': 		[ 32, 32, 'P', 64, 64, 'P', 128, 128, 'P', 256, 256, 'P', 'T', 'P', 256, 256, 'P', 128, 128, 'P', 64, 64, 'P', 32, 3 ],
	'ae_4_resilu': 	[ 32, 32, 'P', 64, 64, 'P', 128, 128, 'P', 256, 256, 'PR', 'ReSiLU', 'T', 'P', 256, 256, 'P', 128, 128, 'P', 64, 64, 'P', 32, 3 ],
	'ae_5_vnn': 	[ 32, 32, 'P', 64, 64, 'P', 128, 128, 'P', 256, 256, 'P', 512, 512, 'PR', 'VNN', 'T', 'P', 512, 512, 'P', 256, 256, 'P', 128, 128, 'P', 64, 64, 'P', 32, 3 ],
	'ae_5_resilu': 	[ 16, 16, 'P', 32, 32, 'P', 64, 64, 'P', 128, 128, 'P', 256, 256, 'PR', 'ReSiLU', 'T', 'P', 256, 256, 'P', 128, 128, 'P', 64, 64, 'P', 32, 32, 'P', 16, 3 ],
}

class VGG_AE(nn.Module, IFixable):
	def __init__(self, cfg_name):
		super(VGG_AE, self).__init__()
		self.features = self._make_layers(ae_cfgs[cfg_name])

		self.hardness_information = -1.0

	def forward(self, x):
		for feature in self.features:
			x = feature(x)
			
		out = x
		return out

	def get_hardness(self) -> float:
		return self.hardness_information
	
	def set_hardness(self, hardness: float):
		for feature in self.features:
			if isinstance(feature, IFixable):
				feature.set_hardness(hardness)
		self.hardness_information = hardness

	def _make_layers(self, config):
		layers = []
		in_channels = 3
		block = 0
		transpose = False

		for x, nextx in zip(config, config[1:]+[None]):
			if x == 'P' or x == 'PR':
				if not transpose:
					layers += [
						nn.AvgPool2d(kernel_size=2, stride=2)
					]
				else:
					layers += [
						nn.Upsample(scale_factor=2)
					]
			elif x == 'T':
				transpose = not transpose
			elif x == 'ReSiLU':
				layers += [
					ReSiLU()
				]
			elif x == 'VNN':
				block += 1
				layers += [
					nn.Flatten(),
					VNN(512, 512, 3, parallel_size=4),
					nn.Unflatten(1, (512, 1, 1)),
					nn.BatchNorm2d(512, track_running_stats=False),
				]
			else:
				if not transpose:
					layers += [
						nn.Conv2d(in_channels, x, kernel_size=3, padding=1)
					]
				else:
					layers += [
						nn.ConvTranspose2d(in_channels, x, kernel_size=3, padding=1)
					]
				layers += [
					nn.ReLU() if nextx != 'PR' else nn.Identity(),
					nn.BatchNorm2d(x, track_running_stats=False),
				]
				in_channels = x

		return nn.ModuleList(layers)

def test():
	net = VGG('vgg19_vnn')
	x = torch.randn(2,3,32,32)
	y = net(x)
	print(y.size())

# test()
