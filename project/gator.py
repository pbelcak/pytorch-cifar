
import torch
from torch import nn

class GateRow(nn.Module):
    def __init__(self, n_gates: int):
        super().__init__()
        self.n_gates = n_gates

        self.gates = nn.Parameter(torch.empty(n_gates, 4, dtype=torch.bool), requires_grad=False)
        self.choices = nn.Parameter(torch.empty(n_gates, 2, dtype=torch.long), requires_grad=False)

    def load(self, gates: torch.Tensor, choices: torch.Tensor):
        self.gates.data = gates
        self.choices.data = choices

    def forward(self, x):
        # input has shape (batch_size, input_width)
        batch_size = x.shape[0]
        
        chosen_inputs = x.index_select(1, self.choices.flatten()).reshape(-1, self.n_gates, 2) # (batch_size, n_gates, 2)

        output_val_indices = chosen_inputs[:, :, 0].int() * 2 + chosen_inputs[:, :, 1].int() # (batch_size, n_gates)
        output_val_indices = output_val_indices + 4*torch.arange(self.n_gates, dtype=torch.long, device=x.device).unsqueeze(0) # (batch_size, n_gates)

        output = self.gates.flatten().index_select(0, output_val_indices.flatten()).reshape(batch_size, self.n_gates) # (batch_size, n_gates)

        return output

class Gator(nn.Module):
    def __init__(self, input_width: int, output_width: int):
        super().__init__()
        self.input_width = input_width
        self.output_width = output_width

        self.gate_rows = nn.ModuleList([])
        self.existing_gates = {}

        self.output_gates = nn.Parameter(torch.zeros(output_width, dtype=torch.long), requires_grad=False)
        self.output_gate_entropies = nn.Parameter(torch.ones(output_width, dtype=torch.float) * 100, requires_grad=False)

    def gate_forward(self, x):
        # input has shape (batch_size, input_width)
        # output has shape (batch_size, output_width)

        for gate_row in self.gate_rows:
            newx = gate_row(x)
            x = torch.cat([x, newx], dim=1)

        return x
    
    def forward(self, x):
        # input has shape (batch_size, input_width)
        # output has shape (batch_size, output_width)

        x = self.gate_forward(x)

        output = x[:, self.output_gates].long()
        return output
    
    def size(self):
        return self.input_width + sum([gate_row.n_gates for gate_row in self.gate_rows])
    
    def find_gate_id(self, ai: int, bi: int):
        return self.existing_gates[(ai, bi)]
    
    def optimization_step(self, x, y, k: int = 1):
        # x has shape (batch_size, self.input_width)
        # y has shape (batch_size, self.output_width)

        x, y = x.bool(), y.bool()

        x = self.gate_forward(x)
        input_width = x.shape[1]
        # x now has shape (batch_size, input_width)

        new_gates = []
        new_functions = []

        for oi in range(self.output_width):
            entropies = torch.ones(input_width, input_width, dtype=torch.float) * 1000.0
            best_function = torch.zeros(input_width, input_width, 4, dtype=torch.bool)
            
            K = 32

            for li in range(min(K, input_width)):
                for si in range(li+1, min(li+K, input_width)):
                    random_indices = torch.randperm(input_width)

                    ai = random_indices[li]
                    bi = random_indices[si]
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

                    # TODO: THIS IS NOT REALLY WORKING AS EXPECTED. WHAT I SHOULD REALLY DO IS TRY ALL POSSIBLE PROJECTIONS AND CHOOSE THE BEST ONE

                    # compute the conditional entropy of trio given duo
                    entropy = 0.0
                    probs = torch.zeros(2, 2, 2, dtype=torch.float)
                    for trio, prob in zip(unique_trios, trio_probs):
                        trio = trio.long()
                        pos = trio[0] * 2 + trio[1]
                        duo_prob = duo_probs[pos]
                        entropy += -prob * torch.log2(prob / duo_prob)
                        probs[trio[0], trio[1], trio[2]] = prob

                    entropies[ai, bi] = entropy
                    best_function[ai, bi] = (probs[:, :, 0] < probs[:, :, 1]).flatten()

            # choose the top k lowest entropy pairs (ai, bi) together with entropy values
            entropies, indices = torch.topk(entropies.flatten(), k, largest=False)
            ai_indices = indices // input_width
            bi_indices = indices % input_width

            # form new gates at these pairs
            for entropy, ai, bi in zip(entropies, ai_indices, bi_indices):
                ai, bi = ai.item(), bi.item()
                #if (ai, bi) not in self.existing_gates:
                best_gate_id = self.size() + len(new_gates) - 1
                new_gates.append([ai, bi])
                new_functions.append(best_function[ai, bi])
                self.existing_gates[(ai, bi)] = best_gate_id
                #else:
                #    best_gate_id = self.find_gate_id(ai, bi)


                # if the newly formed gate is better than self.output_gates[oi], replace it
                if self.output_gate_entropies[oi] > entropy:
                    self.output_gates[oi] = best_gate_id
                    self.output_gate_entropies[oi] = entropy

                
        # create a new gate row from the new_gates double-list
        gate_row = GateRow(len(new_gates))
        gate_row.load(torch.stack(new_functions), torch.tensor(new_gates, dtype=torch.long))

        self.gate_rows.append(gate_row)

        return len(new_gates)

    