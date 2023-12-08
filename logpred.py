import device
import records

import torch
import torch.nn as nn
import wandb

import models
import image
import optimizers

from project.gator import *

from torch.utils.data import TensorDataset

from project.forester import *
	
def get_dataloaders(args):
	# get the data
	hard_representations_training = image.load(args, label='training')
	hard_representations_validation = image.load(args, label='validation')
	hard_representations_testing = image.load(args, label='testing')
	
	# use TensorDataset to create a dataset from hard_representations[0] and hard_representations[1]
	#dataset_training = TensorDataset(hard_representations_training[0], hard_representations_training[1])
	#dataset_validation = TensorDataset(hard_representations_validation[0], hard_representations_validation[1])
	dataset_testing = TensorDataset(hard_representations_testing[0], hard_representations_testing[1])
	
	# use DataLoader to create a dataloader from the dataset
	#dataloader_training = torch.utils.data.DataLoader(dataset_training, batch_size=args.batch_size, shuffle=True)
	#dataloader_validation = torch.utils.data.DataLoader(dataset_validation, batch_size=args.batch_size, shuffle=False)
	dataloader_testing = torch.utils.data.DataLoader(dataset_testing, batch_size=args.batch_size, shuffle=False)

	return dataloader_testing, dataloader_testing, dataloader_testing

@torch.no_grad()
def train(args):
	#hard_representations_training = image.load(args, label='training')
	hard_representations_testing = image.load(args, label='testing')
	#dataset_training = TensorDataset(hard_representations_training[0], hard_representations_training[1])
	dataset_testing = TensorDataset(hard_representations_testing[0][0:1024], hard_representations_testing[1][0:1024])

	dataset_testing = make_unfolded_dataset(args, dataset_testing)

	# create dataloaders for training
	#dataloader_training = torch.utils.data.DataLoader(dataset_training, batch_size=args.batch_size, shuffle=True)
	dataloader_testing = torch.utils.data.DataLoader(dataset_testing, batch_size=args.batch_size, shuffle=True)

	# predictor = Gator(64 * 3 * 3, 128)
	predictor = Forester(64 * 3 * 3, 128, 12)
	for i in range(12):
		# training batch
		data, target = next(iter(dataloader_testing))
		data, target = data.to(device.device), target.to(device.device)
		predictor.make_tree_level(data, target)

		# testing batch
		data, target = next(iter(dataloader_testing))
		data, target = data.to(device.device), target.to(device.device)
		predictor.eval()
		output = predictor(data)
		accuracy = (output == target.bool()).float().mean()
		print(accuracy.item())


def make_unfolded_dataset(args, dataset):
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

	data_acc = []
	targets_acc = []
	for batch_idx, (data, target) in enumerate(dataloader):
		data, target = data.to(device.device), target.to(device.device)

		data = torch.nn.Unfold(kernel_size=(3,3), stride=(2,2), padding=1)(data)
		data = data.transpose(1, 2).round().flatten(0, 1)
		target = target.round().flatten(-2).transpose(1, 2).flatten(0, 1)

		data_acc.append(data.detach().cpu())
		targets_acc.append(target.detach().cpu())
	
	all_data = torch.cat(data_acc)
	all_targets = torch.cat(targets_acc)

	return TensorDataset(all_data, all_targets)


def split_dataset_by_predictor(args, model, dataset):
	model.eval()
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

	lhs_indices = []
	rhs_indices = []
	
	for batch_idx, (data, target) in enumerate(dataloader):
		data, target = data.to(device.device), target.to(device.device)

		output = model(data)
		predictions = (output > 0)
		lhs_indices += [ (batch_idx*args.batch_size + predictions.nonzero(as_tuple=True)[0]).detach().cpu() ]
		rhs_indices += [ (batch_idx*args.batch_size + (~predictions).nonzero(as_tuple=True)[0]).detach().cpu() ]

	lhs_indices = torch.cat(lhs_indices)
	rhs_indices = torch.cat(rhs_indices)

	dataset_lhs = torch.utils.data.Subset(dataset, lhs_indices)
	dataset_rhs = torch.utils.data.Subset(dataset, rhs_indices)

	return dataset_lhs, dataset_rhs


def save(args, model, optimizer, label):
	model_name = models.make_model_name(args, label)
	torch.save(model.state_dict(), models.make_checkpoint_path(args, model_name, component='model'))
	torch.save(optimizer.state_dict(), models.make_checkpoint_path(args, model_name, component='optimizer'))

	return model_name