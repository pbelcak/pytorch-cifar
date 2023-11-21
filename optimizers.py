import torch

def get_optimizer(args, model):
	if args.optimizer == 'adam':
		return torch.optim.Adam(model.parameters(), lr=args.learning_rate)
	elif args.optimizer == 'sgd':
		return torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=0.0) # weight decay 5e-4
	else:
		raise ValueError('Unknown optimizer: %s' % args.optimizer)

def get_scheduler(args, optimizer):
	if args.scheduler == 'cosine':
		return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs*1.05)
	else:
		return None