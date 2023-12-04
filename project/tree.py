import torch
from torch import nn
import math

class Tree(nn.Module):
	def __init__(self, input_width: int, max_depth: int):
		super(Tree, self).__init__()
		self.input_width = input_width
		self.current_depth = 0
		self.max_depth = max_depth
		self.n_nodes = 2 ** max_depth + 1

		self.node_choices = nn.Parameter(torch.empty((self.n_nodes,), dtype=torch.long), requires_grad=False)
		self.node_predictions = nn.Parameter(torch.empty((self.n_nodes,), dtype=torch.bool), requires_grad=False)
		self.node_choices.data.fill_(-1)

	def forward(self, x):
		# input has shape (batch_size, input_width)
		# output has shape (batch_size, 1)

		current_nodes = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
		for d in range(self.current_depth):
			current_choices = self.node_choices[current_nodes]
			decisions = x.gather(1, current_choices.unsqueeze(-1)).squeeze(-1).long()

			current_nodes = current_nodes * 2 + decisions + 1
		
		return self.node_predictions[current_nodes]
	
	def find_best_choice(self, x, y):
		# x has shape (part_size, input_width)
		# y has shape (part_size, 1)
		# output is a pair of (best_choice, best_predictions)

		# compute the conditional entropy H(Y|X) for every choice of position from [0, input_width)
		conditional_entropies = torch.empty((self.input_width,), dtype=torch.float)
		best_predictions = torch.empty((self.input_width, 2), dtype=torch.bool)
		for pos in range(self.input_width):
			xy_pairs = torch.cat([x[:, pos].unsqueeze(-1), y], dim=-1) # shape (part_size, 2)
			xy_counts_vals, xy_counts_counts = torch.unique(xy_pairs, dim=0, sorted=True, return_counts=True)
			xy_counts = torch.zeros((4,), dtype=torch.long, device=x.device)
			
			for xy_pair, xy_count in zip(xy_counts_vals, xy_counts_counts):
				xy_counts[xy_pair[0] * 2 + xy_pair[1]] = xy_count

			xy_probs = xy_counts.float() / xy_counts.sum() # shape (n_unique_xy_pairs,)

			x_is_1_prob = x[:, pos].sum().float() / x.shape[0] # shape (1,)
			x_probs = (1.0 - x_is_1_prob, x_is_1_prob) # shape (2,)

			best_predictions[pos][0] = xy_probs[0] < xy_probs[1]
			best_predictions[pos][1] = xy_probs[2] < xy_probs[3]

			conditional_entropy = 0.0
			for xval in range(2):
				for yval in range(2):
					xy_prob = xy_probs[xval * 2 + yval]
					if xy_prob > 0.0:
						conditional_entropy -= xy_prob * torch.log(xy_prob / x_probs[xval])

			conditional_entropies[pos] = conditional_entropy
			
																	
		# find the best choice
		best_choice = conditional_entropies.argmin()
		return best_choice, best_predictions[best_choice]

	def build(self, x, y, current_node: int = 0, stop_depth: int = None):
		# x has shape (batch_size, input_width)
		# y has shape (part_size, 1)

		if stop_depth is None:
			stop_depth = self.max_depth

		current_depth = math.floor(math.log2(current_node + 1))
		if current_depth+1 > stop_depth:
			return

		if self.node_choices[current_node] == -1:
			# find the best choice
			best_choice, best_predictions = self.find_best_choice(x, y)

			# update the tree
			self.node_choices[current_node] = best_choice
			self.node_predictions[current_node * 2 + 1] = best_predictions[0]
			self.node_predictions[current_node * 2 + 2] = best_predictions[1]
		else:
			best_choice = self.node_choices[current_node]

		leftx_entries = x[:, best_choice] == 0
		rightx_entries = x[:, best_choice] == 1
		leftx = x[leftx_entries]
		lefty = y[leftx_entries]
		self.build(leftx, lefty, current_node * 2 + 1, stop_depth)
		del leftx, lefty

		rightx = x[rightx_entries]
		righty = y[rightx_entries]
		self.build(rightx, righty, current_node * 2 + 2, stop_depth)
		del rightx, righty

		# if the rhs child of the current node exceeds the current number of nodes in the tree, increase the depth
		self.current_depth = max(self.current_depth, current_depth+1)