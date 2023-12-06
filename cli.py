import argparse
import time
import os

def get_args():
	parser = argparse.ArgumentParser(description='Train a model on a dataset.')
	parser.add_argument('--seed',
		type=int,
		default=42,
		help='The seed for all randomness.'
	)
	parser.add_argument('--job_id',
		type=int,
		default=int(time.time()),
		help='The job id for the run.'
	)
	parser.add_argument('--job_suite',
		type=str,
		default="base",
		help='A name used to group a set of jobs together. Is "base" if not specified.'
	)
	parser.add_argument('--action',
		type=str,
		default='train',
		help='action to perform (train, evaluate)'
	)
	parser.add_argument('--dataset',
		type=str,
		default='mnist',
		help='dataset to use (usps, mnist, fashionmnist)'
	)
	parser.add_argument('--split',
		type=str,
		default='testing',
		help='The split to use for evaluation. Can be training, validation, or testing. Will only be used for evaluation.'
	)
	parser.add_argument('--data_directory',
		type=str,
		default='./cache',
		help='Path to where the datasets are located/are to be stored when downloaded for the first time.'
	)
	parser.add_argument('--checkpointing_directory',
		type=str,
		default='./cache/checkpoint',
		help='Path to where the checkpoints are to be loaded from or stored to.'
	)
	parser.add_argument('--logging_directory',
		type=str,
		default='./cache/logging',
		help='Path to where the logs are to be stored.'
	)
	parser.add_argument('--results_directory',
		type=str,
		default='./cache/results',
		help='Path to the directory where the results are to be stored.'
	)
	parser.add_argument('--image_directory',
		type=str,
		default='./cache/images',
		help='Path to the directory where the model images of the input datasets are to be stored.'
	)
	parser.add_argument('--architecture',
		type=str,
		default='ff',
		help='The architecture to use (ff)'
	)
	parser.add_argument('--checkpoint',
		type=str,
		default=None,
		help='The name (conforming to make_model_name() format) of the checkpoint to load (either for continued training or testing). Defaults to None -- nothing will be loaded for training or testing.'
	)
	parser.add_argument('--image',
		type=str,
		default=None,
		help='The name of the representations to load. Defaults to None -- nothing will be loaded.'
	)
	parser.add_argument('--optimizer',
		type=str,
		default='adam',
		help='optimizer to use (adam, sgd)'
	)
	parser.add_argument('--scheduler',
		type=str,
		default=None,
		help='scheduler to use (cosine, default: None)'
	)
	parser.add_argument('--fixation_schedule',
		type=str,
		default=None,
		help='The fixation schedule to use (None, linear_100, linear_60; default: None)'
	)
	parser.add_argument('--fixation_cap',
		type=float,
		default=1.0,
	)
	parser.add_argument('--fixation_flare',
		type=int,
		default=1,
		help='The fixation flare to use (default: 1)'
	)
	parser.add_argument('--max_depth',
		type=int,
		default=2
	)
	parser.add_argument('--learning_rate',
		type=float,
		default=0.001,
		help='learning rate'
	)
	parser.add_argument('--clip',
		type=float,
		default=1.0,
		help='gradient norm clipping (default: 1.0)'
	)
	parser.add_argument('--epochs',
		type=int,
		default=1,
		help='Number of epochs to train for'
	)
	parser.add_argument('--patience',
		type=int,
		default=250,
		help='Number of epochs to wait for improvement before early stopping'
	)
	parser.add_argument('--min_delta',
		type=float,
		default=0.0,
		help='Minimum change in validation loss to qualify as an improvement for the purpose of early stopping'
	)
	parser.add_argument('--batch_size',
		type=int,
		default=256,
		help='batch size'
	)
	parser.add_argument('--evaluate_after_training',
		action='store_true',
		help='Whether to run the evaluation loop after training. Respects the split argument.'
	)
	parser.add_argument('--use_cpu',
		action='store_true',
		help='Overrides the use of CUDA where CUDA is available'
	)

	args = parser.parse_args()

	create_directory_if_does_not_exist(args.data_directory)
	create_directory_if_does_not_exist(args.checkpointing_directory)
	create_directory_if_does_not_exist(args.logging_directory)
	create_directory_if_does_not_exist(args.results_directory)
	create_directory_if_does_not_exist(args.image_directory)

	return args

def create_directory_if_does_not_exist(path):
	if not os.path.exists(path):
		os.makedirs(path)