
import torch
from torch import nn

class Forest(nn.Module):
	def __init__(self, n_trees: int, max_depth: int):
		super().__init__()
		self.n_trees = n_trees
		self.current_depth = 0
		self.max_depth = max_depth
		self.n_nodes = 2 ** max_depth - 1

		self.node_outputs = nn.Parameter(torch.empty(n_trees, self.n_nodes, 2, dtype=torch.bool), requires_grad=False)
		self.node_foci = nn.Parameter(torch.ones(n_trees, self.n_nodes, dtype=torch.long) * -1, requires_grad=False)

	def set_node(self, tree_index: int, node_index, output: torch.Tensor, focus: int):
		self.node_outputs[tree_index, node_index] = output
		self.node_foci[tree_index, node_index] = focus

	def forward(self, x):
		# input has shape (batch_size, input_width)
		batch_size = x.shape[0]

		# do a tree forward pass, saving the intermediate tree level results on the way for reference by further tree rows
		current_nodes = torch.zeros(batch_size, self.n_trees, dtype=torch.long, device=x.device)
		output = None
		for d in range(self.current_depth):
			current_output_maps = self.node_outputs\
				.transpose(-2, -1).unsqueeze(0).expand(batch_size, self.n_trees, 2, self.n_nodes)\
				.gather(3, current_nodes.unsqueeze(-1).unsqueeze(-1).expand(batch_size, self.n_trees, 2, 1))\
				.squeeze(-1) # (batch_size, n_trees, 2)
			current_foci = self.node_foci\
				.unsqueeze(0).expand(batch_size, self.n_trees, self.n_nodes)\
				.gather(2, current_nodes.unsqueeze(-1))\
				.squeeze(-1) # (batch_size, n_trees)
			decisions = x.gather(1, current_foci).long() # (batch_size, n_trees)

			output = current_output_maps.gather(2, decisions.unsqueeze(-1)).squeeze(-1) # (batch_size, n_trees)

			current_nodes = current_nodes * 2 + decisions + 1
			x = torch.cat([x, output], dim=1)

		return x, output

class Forester(nn.Module):
	def __init__(self, input_width: int, n_trees: int, max_depth: int = 5):
		super().__init__()
		self.input_width = input_width
		self.output_width = n_trees

		self.forest = Forest(n_trees, max_depth)
	
	def forward(self, x):
		# input has shape (batch_size, input_width)
		# output has shape (batch_size, output_width)

		aggregate_output, output = self.forest(x)

		return output
	
	def size(self):
		return self.input_width + sum([gate_row.n_gates for gate_row in self.gate_rows])
	
	def make_tree_level(self, x, y):
		# x has shape (batch_size, self.input_width)
		# y has shape (batch_size, self.output_width)

		if self.forest.current_depth == self.forest.max_depth:
			raise Exception("Cannot make a tree level -- the forest is already at max depth")

		x, y = x.bool(), y.bool()

		x, _ = self.forest(x)
		candidate_width = x.shape[1]
		# x now has shape (batch_size, candidate_width)

		for tree_index in range(self.forest.n_trees):
			self.make_children(tree_index, 0, x, y)

		# at the end, increment the self.forest.current_depth
		self.forest.current_depth += 1

	def make_children(self, tree_index, node_index, relevantx, relevanty):
		if self.forest.node_foci[tree_index, node_index] == -1:
			self.make_node(tree_index, node_index, relevantx, relevanty)
			return
		else:
			# split x,y into x0,y0 and x1,y1 depending on the value of the node's focus
			focus_index = self.forest.node_foci[tree_index, node_index]
			zeromap = relevantx[:, focus_index] == 0
			onemap = ~zeromap
			x0, x1 = relevantx[zeromap], relevantx[onemap]
			y0, y1 = relevanty[zeromap], relevanty[onemap]

			# make two new nodes; one for x0,y0 (the left node) and one for x1,y1 (the right node)
			self.make_children(tree_index, node_index * 2 + 1, x0, y0)
			self.make_children(tree_index, node_index * 2 + 2, x1, y1)

	
	def make_node(self, tree_index, node_index, relevantx, relevanty):
		# x has shape (batch_size, self.input_width)
		# y has shape (batch_size, candidate_width)
		x, y = relevantx, relevanty
		candidate_width = x.shape[1]

		marginal_distributions = x.float().sum(dim=0) / x.shape[0]

		candidate_entropies = torch.ones(candidate_width, dtype=torch.float) * 1000.0
		candidate_best_functions = torch.zeros(candidate_width, 2, dtype=torch.bool)
		
		K = 32

		for li in range(min(K, candidate_width)):
				random_indices = torch.randperm(candidate_width)

				ai = random_indices[li]
				duos = torch.cat([
					x[:, ai].unsqueeze(1),
					y[:, tree_index].unsqueeze(1)
				], dim=1)
				# find the distribution of trios (use torch.unique)
				unique_duos, counts = torch.unique(duos, dim=0, return_counts=True)

				xy_counts = torch.zeros((2, 2), dtype=torch.long, device=counts.device)
				count_sum = 0
				for xy_pair, xy_count in zip(unique_duos, counts):
					xy_counts[xy_pair[0].long(), xy_pair[1].long()] = xy_count
					count_sum += xy_count

				xy_probs = xy_counts / count_sum

				# compute the conditional entropy H(Y|X) for every choice of position from [0, candidate_width)
				entropy = 0.0
				for xval in range(2):
					for yval in range(2):
						xy_prob = xy_probs[xval, yval]
						if xy_prob > 1e-10:
							entropy -= xy_prob * torch.log(xy_prob / (marginal_distributions[ai] if xval==1 else 1.0 - marginal_distributions[ai]))

				candidate_entropies[ai] = entropy
				candidate_best_functions[ai, 0] = xy_probs[0, 0] < xy_probs[0, 1]
				candidate_best_functions[ai, 1] = xy_probs[1, 0] < xy_probs[1, 1]
					

		# choose the lowest-entropy predictor from among the candidates
		best_candidate_index = candidate_entropies.argmin().item()
		best_candidate_entropy = candidate_entropies[best_candidate_index]
		best_candidate_function = candidate_best_functions[best_candidate_index] # we choose the best of the best ;)

		# update the tree
		self.forest.set_node(tree_index, node_index, best_candidate_function, best_candidate_index)

	