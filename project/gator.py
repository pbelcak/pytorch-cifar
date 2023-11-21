
import torch
from torch import nn

class GateRow(nn.Module):
    def __init__(self, n_gates: int):
        self.n_gates = n_gates
        self.gates = nn.Parameter(torch.empty(n_gates, 4, dtype=torch.bool))
        self.choices = nn.Parameter(torch.empty(n_gates, 2, dtype=torch.long))

    def load(self, gates: torch.Tensor, choices: torch.Tensor):
        self.gates.data = gates
        self.choices.data = choices

    def forward(self, x):
        # input has shape (batch_size, input_width)
        
        chosen_inputs = x.index_select(1, self.choices.flatten()).reshape(-1, self.n_gates, 2) # (batch_size, n_gates, 2)

        output_val_indices = chosen_inputs[:, :, 0].int() * 2 + chosen_inputs[:, :, 1].int() # (batch_size, n_gates)

        output = self.gates.index_select(1, output_val_indices.transpose(0, 1)).transpose(0, 1) # (batch_size, n_gates)

        return output

class Gator(nn.Module):
    def __init__(self, input_width: int, output_width: int):
        self.input_width = input_width
        self.output_width = output_width

        self.gate_rows = nn.ModuleList([])
        self.existing_gates = set()

        self.output_gates = nn.Parameter(torch.empty(output_width, dtype=torch.long))
        self.output_gate_entropies = nn.Parameter(torch.ones(output_width, dtype=torch.long))

    def forward(self, x):
        # input has shape (batch_size, input_width)
        # output has shape (batch_size, output_width)

        for gate_row in self.gate_rows:
            newx = gate_row(x)
            x = torch.cat([x, newx], dim=1)

        return x
    
    def optimization_step(self, x, y, k: int = 1):
        # x has shape (batch_size, self.input_width)
        # y has shape (batch_size, self.output_width)

        x = self(x)
        input_width = x.shape[1]
        # x now has shape (batch_size, input_width)

        new_gates = []
        new_functions = []

        for oi in range(self.output_width):
            entropies = torch.empty(input_width, input_width, dtype=torch.float)
            best_function = torch.zero(input_width, input_width, 4, dtype=torch.bool)
            
            for ai in range(input_width):
                for bi in range(input_width):
                    trios = torch.cat([
                        x[:, ai].unsqueeze(1),
                        x[:, bi].unsqueeze(1),
                        y[:, oi].unsqueeze(1)
                    ], dim=1)
                    # find the distribution of trios (use torch.unique)
                    unique_trios, counts = torch.unique(trios, dim=0, return_counts=True)
                    trio_probs = counts / torch.sum(counts)

                    # compute the marginal distribution of the trios using the first two columns
                    duo_counts = [0, 0, 0, 0]
                    for trio, count in zip(unique_trios, counts):
                        duo_counts[trio[0] * 2 + trio[1]] += count

                    # turn counts into probabilities
                    duo_sum = sum(duo_counts)
                    duo_probs = [count / duo_sum for count in duo_counts]

                    # compute the conditional entropy of trio given duo
                    entropy = 0.0
                    probs = torch.zero(2, 2, 2, dtype=torch.float)
                    for trio, prob in zip(unique_trios, trio_probs):
                        pos = trio[0] * 2 + trio[1]
                        duo_prob = duo_probs[pos]
                        entropy += -prob * torch.log2(prob / duo_prob)
                        probs[trio[0], trio[1], trio[2]] = prob

                    entropies[ai, bi] = entropy
                    best_function[ai, bi] = (probs[:, :, 0] > probs[:, :, 1]).flatten()

            # choose the top k lowest entropy pairs (ai, bi) together with entropy values
            entropies, indices = torch.topk(entropies.flatten(), k, largest=False)
            ai_indices = indices // self.input_width
            bi_indices = indices % self.input_width

            # form new gates at these pairs
            for entropy, ai, bi in zip(entropies, ai_indices, bi_indices):
                if (ai, bi) not in self.existing_gates:
                    new_gates.append([ai, bi])
                    new_functions.append(best_function[ai, bi])
                    self.existing_gates.add((ai, bi))

                
        # create a new gate row from the new_gates double-list
        gate_row = GateRow(len(new_gates))
        gate_row.load(torch.stack(new_functions), torch.tensor(new_gates, dtype=torch.long))

        self.gate_rows.append(gate_row)

        return len(new_gates)

    