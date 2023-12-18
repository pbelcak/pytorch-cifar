import data
import device
import records

import torch
import torch.nn as nn
import wandb
import os

import models

def compute_loss_and_accuracy(output, target):
	loss = nn.CrossEntropyLoss(reduction='none')(output, target)
	accuracy = (output.argmax(dim=1) == target).float()
	return loss, accuracy

def choose_evaluation_dataloader(args, training, validation, testing):
	if args.split == 'training':
		return training
	elif args.split == 'validation':
		return validation
	elif args.split == 'testing':
		return testing
	else:
		raise ValueError('Unknown split: %s' % args.split)

def train(args, model, optimizer, scheduler=None):
    # get the data
	dataloader_training, dataloader_validation, dataloader_testing = data.get_dataloaders(args)

	best_validation_loss = float('inf')
	best_validation_loss_epoch = 0

	records.update_job_status(args, 'training')

	total_steps = len(dataloader_training) * args.epochs
	hardening_steps = 0
	if args.fixation_schedule is not None:
		if args.fixation_schedule == 'linear_100':
			hardening_steps = total_steps*1.005
		elif args.fixation_schedule == 'cubic_100':
			hardening_steps = total_steps*1.005
		elif args.fixation_schedule == 'linear_60':
			hardening_steps = int(total_steps * 0.60)
		else:
			raise ValueError('Unknown fixation schedule: %s' % args.fixation_schedule)

	# train the model
	for epoch in range(args.epochs):
		elapsed_steps = epoch * len(dataloader_training)
		print('Epoch %d/%d' % (epoch+1, args.epochs))

		#if epoch > 0.60*args.epochs:
			#for param_group in optimizer.param_groups:
				#param_group['weight_decay'] = 0.0
				#param_group['momentum'] = 0.0

		# training
		training_result = loop(args, model, 'training', dataloader_training, optimizer, elapsed_steps, hardening_steps)
	
		# validation
		soft_validation_result = loop(args, model, 'soft_validation', dataloader_testing)

		# validation
		hard_validation_result = loop(args, model, 'hard_validation', dataloader_testing)
		wandb.log({
			'epoch': epoch,
			'training_loss': training_result['loss'],
			'training_accuracy': training_result['accuracy'],
			'soft_validation_loss': soft_validation_result['loss'],
			'soft_validation_accuracy': soft_validation_result['accuracy'],
			'hard_validation_loss': hard_validation_result['loss'],
			'hard_validation_accuracy': hard_validation_result['accuracy']
		})

		# early stopping and checkpointing
		if best_validation_loss == float('inf') or hard_validation_result['loss'] < best_validation_loss - (best_validation_loss * args.min_delta):
			best_validation_loss = hard_validation_result['loss']
			best_validation_loss_epoch = epoch
			#best_model_name = save(args, model, optimizer, 'best')
			#records.insert_model(
			#	args, best_model_name, epoch+1,
			#	training_result['loss'], training_result['accuracy'], hard_validation_result['loss'], hard_validation_result['accuracy']
			#)
		elif epoch - best_validation_loss_epoch >= args.patience:
			break


		if scheduler is not None:
			scheduler.step()

	# save the last checkpoint
	last_model_name = save(args, model, optimizer, 'last')
	records.insert_model(
		args, last_model_name, epoch+1,
		training_result['loss'], training_result['accuracy'], hard_validation_result['loss'], hard_validation_result['accuracy']
	)

	result = {
		'epochs': epoch,
		'training_loss': training_result['loss'],
		'training_accuracy': training_result['accuracy'],
		'validation_loss': hard_validation_result['loss'],
		'validation_accuracy': hard_validation_result['accuracy'],
		'best_validation_loss': best_validation_loss,
		'best_validation_loss_epoch': best_validation_loss_epoch
	}

	records.update_job_status(args, 'trained')

	if args.evaluate_after_training:
		dataloader_evaluation = choose_evaluation_dataloader(args, dataloader_training, dataloader_validation, dataloader_testing)
		evaluation_result = evaluate(args, last_model_name, model, dataloader_evaluation)
		result = {**result, **evaluation_result}

	return result

def evaluate(args, model_name, model, dataloader=None):
	if dataloader is None:
		training, validation, testing = data.get_dataloaders(args)
		dataloader = choose_evaluation_dataloader(args, training, validation, testing)

	# run the evaluation loop
	records.update_job_status(args, 'evaluating')
	evaluation_result = loop(args, model, 'evaluation', dataloader)
	records.update_job_status(args, 'evaluated')
	records.insert_evaluation(args, model_name, evaluation_result['loss'], evaluation_result['accuracy'])

	# return an informative result dictionary
	return {
		'evaluation_loss': evaluation_result['loss'],
		'evaluation_accuracy': evaluation_result['accuracy']
	}

def loop(args, model, mode, dataloader, optimizer=None, epoch_elapsed_steps=0, hardening_steps=0):
	if mode == 'training' or mode == 'soft_validation':
		model.train()
	elif mode == 'validation' or mode == 'hard_validation' or mode == 'evaluation':
		model.eval()

	loss_accumulator = 0.0
	accuracy_accumulator = 0.0
	accumulator_count = 0
	for batch_idx, (data, target) in enumerate(dataloader):
		elapsed_steps = epoch_elapsed_steps + batch_idx
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
			model.set_hardness(hardness)

		data, target = data.to(device.device), target.to(device.device)

		output = model(data)
		batch_size = output.shape[0]
		loss, accuracy = compute_loss_and_accuracy(output, target)

		loss_accumulator += loss.sum().item()
		accuracy_accumulator += accuracy.sum().item()
		accumulator_count += batch_size

		mean_loss = loss.mean()
		mean_accuracy = accuracy.mean()

		if optimizer != None:
			mean_loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
			optimizer.step()

		if mode == 'training':
			wandb.log({
				'batch_training_loss': mean_loss.item(),
				'batch_training_accuracy': mean_accuracy.item(),
				'hardness': model.get_hardness()
			})

	total_mean_loss = loss_accumulator / accumulator_count
	total_mean_accuracy = accuracy_accumulator / accumulator_count

	return {
		'loss': total_mean_loss,
		'accuracy': total_mean_accuracy
	}

def save(args, model, optimizer, label):
	model_name = models.make_model_name(args, label)
	torch.save(model.state_dict(), models.make_checkpoint_path(args, model_name, component='model'))
	torch.save(optimizer.state_dict(), models.make_checkpoint_path(args, model_name, component='optimizer'))

	return model_name