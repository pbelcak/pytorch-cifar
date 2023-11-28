import models
import data

def build(args, model):
	'''
	Inputs: dataset, model.
	Outputs: in the filesystem, a file storing a tensor representation of the dataset samples as they are fed through the model.
	'''

	dataloader_training, dataloader_validation, dataloader_testing = data.get_dataloaders(args)

	
	pass