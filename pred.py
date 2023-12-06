import device
import records

import torch
import torch.nn as nn
import wandb

import models
import image
import optimizers

from torch.utils.data import TensorDataset

def compute_loss_and_accuracy(output, target):
	loss = torch.nn.functional.binary_cross_entropy_with_logits(output, target, reduction='none')
	accuracy = ((output > 0) == target.bool()).float()
	return loss, accuracy
	
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

def train(args):
	hard_representations_training = image.load(args, label='training')
	hard_representations_testing = image.load(args, label='testing')
	dataset_training = TensorDataset(hard_representations_training[0], hard_representations_training[1])
	dataset_testing = TensorDataset(hard_representations_testing[0][0:1024], hard_representations_testing[1][0:1024])

	dataset_testing = make_unfolded_dataset(args, dataset_testing)

	predictors = train_node(args, 0, dataset_testing, dataset_testing)
	return {}
		

def train_node(args, node_id: int, dataset_training: TensorDataset, dataset_testing: TensorDataset):
	# create a new instance of the predictor model
	predictor = models.get_model(args, None)
	predictor = predictor.to(device.device)

	ret = { node_id: predictor }

	# train the predictor
	predictor_training_result = predictor_train(args, f"{node_id}", predictor, dataset_training, dataset_testing)
	wandb.log({
		f"{node_id}/training_accuracy": predictor_training_result['training_accuracy'],
		f"{node_id}/validation_accuracy": predictor_training_result['validation_accuracy'],
	})

	if node_id > 2 ** (args.max_depth - 1):
		return ret

	# split dataset_training, dataset_testing into two new datasets according to the prediction of the predictor for each sample
	predictor.eval()
	dataset_training_0, dataset_training_1 = split_dataset_by_predictor(args, predictor, dataset_training)
	dataset_testing_0, dataset_testing_1 = split_dataset_by_predictor(args, predictor, dataset_testing)
	# del dataset_training, dataset_testing

	# if the dataset_training_0 or dataset_training_1 is empty, return the predictor
	if len(dataset_training_0) == 0 or len(dataset_training_1) == 0:
		return ret

	# train the two new nodes
	left_node_id = node_id * 2 + 1
	right_node_id = node_id * 2 + 2
	ret.update(train_node(args, left_node_id, dataset_training_0, dataset_testing_0))
	ret.update(train_node(args, right_node_id, dataset_training_1, dataset_testing_1))

	return ret


def predictor_train(args, name, model, dataset_training, dataset_testing):
	# create dataloaders for training
	dataloader_training = torch.utils.data.DataLoader(dataset_training, batch_size=args.batch_size, shuffle=True)
	dataloader_testing = torch.utils.data.DataLoader(dataset_testing, batch_size=args.batch_size, shuffle=False)

	# create a new instance of the optimizer and scheduler
	optimizer = optimizers.get_optimizer(args, model)
	scheduler = optimizers.get_scheduler(args, optimizer)

	total_steps = 40_000
	hardening_steps = 0
	if args.fixation_schedule is not None:
		if args.fixation_schedule == 'linear_100':
			hardening_steps = total_steps*1.01
		elif args.fixation_schedule == 'cubic_100':
			hardening_steps = total_steps*1.01
		elif args.fixation_schedule == 'linear_60':
			hardening_steps = int(total_steps * 0.605)
		else:
			raise ValueError('Unknown fixation schedule: %s' % args.fixation_schedule)

	# train the model
	epoch = 0
	while True:
		elapsed_steps = epoch * len(dataloader_training)
		if elapsed_steps > total_steps:
			break

		# training
		training_result = predictor_loop(args, model, 'training', dataloader_training, optimizer, elapsed_steps, hardening_steps)
	
		wandb.log({
			name+'/epoch': epoch,
			name+'/training_loss': training_result['loss'],
			name+'/training_accuracy': training_result['accuracy']
		})

		epoch += 1

		if scheduler is not None:
			scheduler.step()

	# run the final validation
	validation_result = predictor_loop(args, model, 'hard_validation', dataloader_testing)

	result = {
		'epochs': epoch,
		'training_loss': training_result['loss'],
		'training_accuracy': training_result['accuracy'],
		'validation_loss': validation_result['loss'],
		'validation_accuracy': validation_result['accuracy'],
	}

	return result

def predictor_loop(args, model, mode, dataloader, optimizer=None, epoch_elapsed_steps=0, hardening_steps=0):
	if mode == 'training' or mode == 'soft_validation':
		model.train()
	elif mode == 'validation' or mode == 'hard_validation' or mode == 'evaluation':
		model.eval()

	loss_accumulator = 0.0
	accuracy_accumulator = 0.0
	accumulator_count = 0
	for batch_idx, (data, target) in enumerate(dataloader):
		elapsed_steps = epoch_elapsed_steps + batch_idx
		if mode == 'training' and elapsed_steps > hardening_steps and hardening_steps > 0:
			break
		accumulator_count += 1
		break

		if optimizer is not None:
			optimizer.zero_grad()

		if optimizer is not None and hardening_steps > 0:
			if args.fixation_schedule.startswith('linear'):
				hardness = min(1.0, elapsed_steps / (hardening_steps + 1))
			elif args.fixation_schedule.startswith('cubic'):
				hardness = (1 - (1 - elapsed_steps / (hardening_steps + 1)) ** 3)
			else:
				hardness = 0.0
			hardness = min(args.fixation_cap, hardness)
			if isinstance(model, models.IFixable):
				model.set_hardness(hardness)

		data, target = data.to(device.device), target.to(device.device)

		#data = torch.nn.Unfold(kernel_size=(10,10), stride=(2,2), padding=4)(data)
		#data = data.transpose(1, 2).round().flatten(0, 1)
		#target = target.round().flatten(-2).transpose(1, 2).flatten(0, 1)[:, 0]

		output = model(data)
		loss, accuracy = compute_loss_and_accuracy(output, target)

		loss_accumulator += loss.sum().item()
		accuracy_accumulator += accuracy.sum().item()
		accumulator_count += output.numel()

		mean_loss = loss.mean()
		mean_accuracy = accuracy.mean()

		if optimizer != None:
			mean_loss.backward()
			# torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
			optimizer.step()

		if mode == 'training':
			wandb.log({
				'batch_training_loss': mean_loss.item(),
				'batch_training_accuracy': mean_accuracy.item(),
				'hardness': model.get_hardness() if isinstance(model, models.IFixable) else 0.0,
			})

	total_mean_loss = loss_accumulator / accumulator_count
	total_mean_accuracy = accuracy_accumulator / accumulator_count

	return {
		'loss': total_mean_loss,
		'accuracy': total_mean_accuracy
	}

def make_unfolded_dataset(args, dataset):
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

	data_acc = []
	targets_acc = []
	for batch_idx, (data, target) in enumerate(dataloader):
		data, target = data.to(device.device), target.to(device.device)

		data = torch.nn.Unfold(kernel_size=(10,10), stride=(2,2), padding=4)(data)
		data = data.transpose(1, 2).round().flatten(0, 1)
		target = target.round().flatten(-2).transpose(1, 2).flatten(0, 1)[:, 0]

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