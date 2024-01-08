import cli
import device
import data
import models
import optimizers
import loops
import aeloop
import records
import image

import torch
import torch.nn as nn
import random
import numpy as np
import os

import wandb

PROJECT_NAME = 'pytorch-cifar'

def main():
	# initialize a wandb run
	args = cli.get_args()
	wandb.init(
		project=PROJECT_NAME if not args.architecture.startswith('ae') else 'cifar10-ae',
		name="{}-{}".format(args.job_id, args.job_suite),
		tags=[args.action, args.dataset, args.architecture],
		config=args,
		dir=args.logging_directory
	)

	# set the seed
	torch.manual_seed(args.seed)
	random.seed(args.seed)
	np.random.seed(args.seed)

	# decide on device
	device.decide_on_device(args)

	# get the data meta
	data_meta = data.get_data_meta(args)

	# get the model
	model = models.get_model(args, data_meta)
	model = model.to(device.device)

	# get the optimizer
	optimizer = optimizers.get_optimizer(args, model)
	scheduler = optimizers.get_scheduler(args, optimizer)

	# if the checkpoint has been specified, load it
	if args.checkpoint != None:
		full_checkpoint_path_model = os.path.join(args.checkpointing_directory, args.checkpoint + '-model.pt')
		# full_checkpoint_path_optimizer = os.path.join(args.checkpointing_directory, args.checkpoint + '-optimizer.pt')
		model.load_state_dict(torch.load(full_checkpoint_path_model))
		# optimizer.load_state_dict(torch.load(full_checkpoint_path_optimizer, map_location=device.device))

	# register the job
	records.initialize(args, project_name=PROJECT_NAME)
	records.insert_job(args)

	# run the action requested
	if args.action == 'train':
		if args.architecture.startswith('ae'):
			result = aeloop.train(args, model, optimizer, scheduler)
		else:
			result = loops.train(args, model, optimizer, scheduler)
	elif args.action == 'evaluate':
		result = loops.evaluate(args, args.checkpoint, model)
	elif args.action == 'build_image':
		result = image.build_all(args, model)
	elif args.action == 'fit_image':
		import popcnt
		result = popcnt.train(args, model, optimizer, scheduler)
	elif args.action == 'entcnt':
		import entcnt
		result = entcnt.train(args, model)
	elif args.action == 'pred':
		import pred
		result = pred.train(args)
	elif args.action == 'logpred':
		import logpred
		result = logpred.train(args)
	else:
		raise ValueError('Unknown action: %s' % args.action)

	# log and print the result
	wandb.log(result)
	print(f"Action {args.action} performed, result:")
	for key, value in result.items():
		print(f"{key}: {value}")

	records.update_job_status(args, 'finished')

if __name__ == '__main__':
	main()