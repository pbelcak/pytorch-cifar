import device
import records

import torch
import torch.nn as nn
import wandb

import models
import image

from torch.utils.data import TensorDataset

def compute_accuracy(output, target):
	accuracy = ((output > 0) == target.bool()).float().mean()
	return accuracy

def choose_evaluation_dataloader(args, training, validation, testing):
	if args.split == 'training':
		return training
	elif args.split == 'validation':
		return validation
	elif args.split == 'testing':
		return testing
	else:
		raise ValueError('Unknown split: %s' % args.split)
	
def get_dataloaders(args):
	# get the data
	hard_representations_training = image.load(args, label='training')
	hard_representations_validation = image.load(args, label='validation')
	hard_representations_testing = image.load(args, label='testing')
	
	# use TensorDataset to create a dataset from hard_representations[0] and hard_representations[1]
	dataset_training = TensorDataset(hard_representations_training[0], hard_representations_training[1])
	dataset_validation = TensorDataset(hard_representations_validation[0], hard_representations_validation[1])
	dataset_testing = TensorDataset(hard_representations_testing[0], hard_representations_testing[1])
	
	# use DataLoader to create a dataloader from the dataset
	dataloader_training = torch.utils.data.DataLoader(dataset_training, batch_size=args.batch_size, shuffle=True)
	dataloader_validation = torch.utils.data.DataLoader(dataset_validation, batch_size=args.batch_size, shuffle=False)
	dataloader_testing = torch.utils.data.DataLoader(dataset_testing, batch_size=args.batch_size, shuffle=False)

	return dataloader_training, dataloader_validation, dataloader_testing

@torch.no_grad()
def train(args, model):
	dataloader_training, dataloader_validation, dataloader_testing = get_dataloaders(args)

	records.update_job_status(args, 'training')

	# train the model
	#for depth in range(10):
	for batch_idx, (data, target) in enumerate(dataloader_testing):
		data, target = data.to(device.device), target.to(device.device)

		data = torch.nn.Unfold(kernel_size=(5,5), stride=(2,2), padding=2)(data)
		data = data.transpose(1, 2).round().flatten(0, 1).bool()
		target = target.round().flatten(-2).transpose(1, 2).flatten(0, 1)[:, 0:1].bool()

		model.build(data, target, stop_depth=9)

		output = model(data).unsqueeze(-1)
		accuracy = compute_accuracy(output, target)
		break

	result = {

	}

	records.update_job_status(args, 'trained')

	return result