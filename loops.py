import data
import device
import records

import torch
import torch.nn as nn
import wandb
import os

import models

def compute_loss_and_accuracy(output, target):
	loss = nn.CrossEntropyLoss()(output, target)
	accuracy = (output.argmax(dim=1) == target).float().mean()
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
			hardening_steps = total_steps
		elif args.fixation_schedule == 'linear_60':
			hardening_steps = int(total_steps * 0.60)
		else:
			raise ValueError('Unknown fixation schedule: %s' % args.fixation_schedule)

	# train the model
	for epoch in range(args.epochs):
		elapsed_steps = epoch * len(dataloader_training)

		# if epoch > 0.50*args.epochs:
		#	for param_group in optimizer.param_groups:
		#		param_group['weight_decay'] = 0.0
		#		param_group['momentum'] = 0.0

		# training
		training_result = loop(args, model, 'training', dataloader_training, optimizer, elapsed_steps, hardening_steps)
	
		# validation
		validation_result = loop(args, model, 'validation', dataloader_testing)
		wandb.log({
			'epoch': epoch,
			'training_loss': training_result['loss'],
			'training_accuracy': training_result['accuracy'],
			'validation_loss': validation_result['loss'],
			'validation_accuracy': validation_result['accuracy']
		})

		# early stopping and checkpointing
		if best_validation_loss == float('inf') or validation_result['loss'] < best_validation_loss - (best_validation_loss * args.min_delta):
			best_validation_loss = validation_result['loss']
			best_validation_loss_epoch = epoch
			best_model_name = save(args, model, optimizer, 'best')
			records.insert_model(
				args, best_model_name, epoch+1,
				training_result['loss'], training_result['accuracy'], validation_result['loss'], validation_result['accuracy']
			)
		elif epoch - best_validation_loss_epoch >= args.patience:
			break


		if scheduler is not None:
			scheduler.step()

	# save the last checkpoint
	last_model_name = save(args, model, optimizer, 'last')
	records.insert_model(
		args, last_model_name, epoch+1,
		training_result['loss'], training_result['accuracy'], validation_result['loss'], validation_result['accuracy']
	)

	result = {
		'epochs': epoch,
		'training_loss': training_result['loss'],
		'training_accuracy': training_result['accuracy'],
		'validation_loss': validation_result['loss'],
		'validation_accuracy': validation_result['accuracy'],
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
	if mode == 'training':
		model.train()
	elif mode == 'validation' or mode == 'evaluation':
		model.eval()

	loss_accumulator = 0.0
	accuracy_accumulator = 0.0
	accumulator_count = 0
	for batch_idx, (data, target) in enumerate(dataloader):
		elapsed_steps = epoch_elapsed_steps + batch_idx
		if optimizer is not None:
			optimizer.zero_grad()

		if optimizer is not None and hardening_steps > 0:
			hardness = min(1.0, elapsed_steps / (hardening_steps + 1))
			model.set_hardness(hardness)

		data, target = data.to(device.device), target.to(device.device)

		output = model(data)
		loss, accuracy = compute_loss_and_accuracy(output, target)

		if optimizer != None:
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
			optimizer.step()

		if mode == 'training':
			wandb.log({
				'batch_training_loss': loss.item(),
				'batch_training_accuracy': accuracy.item(),
				'hardness': model.get_hardness()
			})

		loss_accumulator += loss.item()
		accuracy_accumulator += accuracy.item()
		accumulator_count += output.shape[0]

	mean_loss = loss_accumulator / batch_idx
	mean_accuracy = accuracy_accumulator / batch_idx

	return {
		'loss': mean_loss,
		'accuracy': mean_accuracy
	}

def save(args, model, optimizer, label):
	model_name = models.make_model_name(args, label)
	torch.save(model.state_dict(), models.make_checkpoint_path(args, model_name, component='model'))
	torch.save(optimizer.state_dict(), models.make_checkpoint_path(args, model_name, component='optimizer'))

	return model_name