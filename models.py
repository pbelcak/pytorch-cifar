import torch
import os

from architectures import *
import project

def make_model_name(args, label):
	return f"{args.job_id}-{args.job_suite}-{label}"

def make_checkpoint_path(args, model_name: str, component: str = 'model'):
    return os.path.join(args.checkpointing_directory, f"{model_name}-{component}.pt")

def get_model(args, data_meta):
	if args.architecture.startswith('vgg'):
		return VGG(args.architecture)
	elif args.architecture.startswith('ae'):
		return VGG_AE(args.architecture)
	elif args.architecture == 'resnet18':
		return ResNet18()
	elif args.architecture == 'preactresnet18':
		return PreActResNet18()
	elif args.architecture == 'googlenet':
		return GoogLeNet()
	elif args.architecture == 'densenet121':
		return DenseNet121()
	elif args.architecture == 'resnext29_2x64d':
		return ResNeXt29_2x64d()
	elif args.architecture == 'mobilenet':
		return MobileNet()
	elif args.architecture == 'mobilenetv2':
		return MobileNetV2()
	elif args.architecture == 'dpn92':
		return DPN92()
	elif args.architecture == 'shufflenetg2':
		return ShuffleNetG2()
	elif args.architecture == 'senet18':
		return SENet18()
	elif args.architecture == 'shufflenetv2':
		return ShuffleNetV2(1)
	elif args.architecture == 'efficientnetb0':
		return EfficientNetB0()
	elif args.architecture == 'regnetx_200mf':
		return RegNetX_200MF()
	elif args.architecture == 'simpledla':
		return SimpleDLA()
	elif args.architecture.startswith('ff'):
		return FF(data_meta['input_width'], data_meta['output_width'], args.architecture)
	elif args.architecture == 'popcnt':
		return Popcnt(data_meta['input_width'], data_meta['output_width'])
	elif args.architecture == 'difflogic-5':
		from difflogic import LogicLayer, GroupSum
		return torch.nn.Sequential(
			torch.nn.Flatten(),
			LogicLayer(64*5*5, 16*64*64, device='cuda', implementation='cuda', grad_factor=2),
			LogicLayer(16*64*64, 16*64*64, device='cuda', implementation='cuda', grad_factor=2),
			LogicLayer(16*64*64, 16*64*64, device='cuda', implementation='cuda', grad_factor=2),
			LogicLayer(16*64*64, 16*64*64, device='cuda', implementation='cuda', grad_factor=2),
			LogicLayer(16*64*64, 16*64*64, device='cuda', implementation='cuda', grad_factor=2),
			GroupSum(k=128, tau=32)
		)
	elif args.architecture == 'difflogic-7':
		from difflogic import LogicLayer, GroupSum
		return torch.nn.Sequential(
			torch.nn.Flatten(),
			LogicLayer(64*5*5, 16*64*64, device='cuda', implementation='cuda', grad_factor=2),
			LogicLayer(16*64*64, 16*64*64, device='cuda', implementation='cuda', grad_factor=2),
			LogicLayer(16*64*64, 16*64*64, device='cuda', implementation='cuda', grad_factor=2),
			LogicLayer(16*64*64, 16*64*64, device='cuda', implementation='cuda', grad_factor=2),
			LogicLayer(16*64*64, 16*64*64, device='cuda', implementation='cuda', grad_factor=2),
			LogicLayer(16*64*64, 16*64*64, device='cuda', implementation='cuda', grad_factor=2),
			LogicLayer(16*64*64, 16*64*64, device='cuda', implementation='cuda', grad_factor=2),
			GroupSum(k=128, tau=32)
		)
	elif args.architecture == 'difflogic-3':
		from difflogic import LogicLayer, GroupSum
		return torch.nn.Sequential(
			torch.nn.Flatten(),
			LogicLayer(64*5*5, 64*16*16, device='cuda', implementation='cuda', grad_factor=1),
			LogicLayer(64*16*16, 64*16*16, device='cuda', implementation='cuda', grad_factor=1),
			LogicLayer(64*16*16, 64*16*16, device='cuda', implementation='cuda', grad_factor=1),
			GroupSum(k=128, tau=8)
		)
	elif args.architecture == 'diffsanity':
		return torch.nn.Sequential(
			torch.nn.Flatten(),
			nn.Linear(64*5*5, 3200),
			nn.ReLU(),
			nn.LayerNorm(3200),
			nn.Linear(3200, 1600),
			nn.ReLU(),
			nn.LayerNorm(1600),
			nn.Linear(1600, 128)
		)
	elif args.architecture == 'tree':
		return project.Tree(64 * 5 * 5, 10)
	elif args.architecture == 'densesumunit':
		return DenseSumUnit(64 * 10 * 10)
	elif args.architecture == 'q1ff':
		return Q1FF(data_meta['input_width'], data_meta['output_width'])
	elif args.architecture == 's1ff':
		return S1FF(data_meta['input_width'], data_meta['output_width'])
	elif args.architecture == 'classicff':
		return ClassicFF(data_meta['input_width'], data_meta['output_width'])
	elif args.architecture == 'resiq1ff':
		return ResiQ1FF(data_meta['input_width'], data_meta['output_width'])
	else:
		raise ValueError('Unknown architecture: %s' % args.architecture)
