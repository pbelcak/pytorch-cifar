import models
import data

import torch
import os
import device

from torch.utils.data import TensorDataset

def build_all(args, model):
	dataloader_training, dataloader_validation, dataloader_testing = data.get_dataloaders(args)

	build(args, model, dataloader_training, label='training')
	build(args, model, dataloader_validation, label='validation')
	build(args, model, dataloader_testing, label='testing')

	return {}

def build(args, model, dataloader, label='training'):
	model.eval()
	model.clean_hard_tensors()
	
	hard_representations = []
	for batch_idx, (inputs, targets) in enumerate(dataloader):
		inputs, targets = inputs.to(device.device), targets.to(device.device)
		_ = model(inputs)

		hard_tensors = model.get_hard_tensors()
		cpu_hard_tensors = []
		model.clean_hard_tensors()
		for hard_tensor in hard_tensors:
			hard_tensor = hard_tensor.detach().cpu()
			cpu_hard_tensors.append(hard_tensor)
		
		# check if hard_representations is empty
		if len(hard_representations) == 0:
			# if yes, initialize hard_representations with the cpu_hard_tensors
			hard_representations = cpu_hard_tensors
		else:
			# if not, concatenate hard_representations with the cpu_hard_tensors
			hard_representations = [torch.cat((hard_representations[i], cpu_hard_tensors[i]), dim=0) for i in range(len(hard_representations))]

	# save the hard_representations
	name = f"{args.job_id}-{args.job_suite}-{args.dataset}-{args.architecture}-{label}"
	torch.save(hard_representations, os.path.join(args.image_directory, name + '.pt'))

def load(args, label='training'):
	name = f"{args.image}-{label}"
	hard_representations = torch.load(os.path.join(args.image_directory, name + '.pt'))
	return hard_representations