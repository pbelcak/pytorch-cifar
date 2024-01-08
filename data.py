import torch
from torchvision import datasets, transforms

def get_data_meta(args):
	if args.dataset in data_meta:
		return data_meta[args.dataset]
	else:
		raise ValueError('Unknown dataset: %s' % args.dataset)

def get_dataloaders(args):
	if args.dataset == 'mnist':
		return get_dataloaders_mnist(args)
	elif args.dataset == 'cifar10':
		return get_dataloaders_cifar10(args)
	else:
		raise ValueError('Unknown dataset: %s' % args.dataset)
	
data_meta = {
	'mnist': {
		'input_width': 28 * 28 * 1,
		'output_width': 10
	},
	'cifar10': {
		'input_width': 32 * 32 * 3,
		'output_width': 10
	},
}
	
def get_dataloaders_mnist(args):
	transform = transforms.Compose([transforms.ToTensor() ])
	
	# TRAINING
	dataset_training = datasets.MNIST(args.data_directory, download=True, train=True, transform=transform)
	training_size = int(0.9 * len(dataset_training))
	split_training, split_validation = torch.utils.data.random_split(dataset_training, [training_size, len(dataset_training) - training_size])

	dataloader_training = torch.utils.data.DataLoader(split_training, batch_size=args.batch_size, shuffle=True)
	dataloader_validation = torch.utils.data.DataLoader(split_validation, batch_size=args.batch_size, shuffle=True)
	
	# TESTING
	dataset_testing = datasets.MNIST(args.data_directory, download=True, train=False, transform=transform)
	dataloader_testing = torch.utils.data.DataLoader(dataset_testing, batch_size=args.batch_size, shuffle=False)

	return dataloader_training, dataloader_validation, dataloader_testing

def get_dataloaders_cifar10(args):
	transform_train = transforms.Compose([
		transforms.RandomCrop(32, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])

	transform_test = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])

	dataset_training = datasets.CIFAR10(
		root=args.data_directory, train=True, download=True, transform=transform_train)
	training_size = int(0.95 * len(dataset_training))
	# split_training, split_validation = torch.utils.data.random_split(dataset_training, [training_size, len(dataset_training) - training_size])
	
	dataloader_training = torch.utils.data.DataLoader(dataset_training, batch_size=args.batch_size, shuffle=True, num_workers=16)
	# dataloader_validation = torch.utils.data.DataLoader(split_validation, batch_size=args.batch_size, shuffle=True)

	dataset_testing = datasets.CIFAR10(args.data_directory, download=True, train=False, transform=transform_test)
	dataloader_testing = torch.utils.data.DataLoader(dataset_testing, batch_size=args.batch_size, shuffle=False, num_workers=16)

	# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

	return dataloader_training, dataloader_testing, dataloader_testing

	