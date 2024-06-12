import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionBasedGating(nn.Module):
    def __init__(self, dim, num_gates, num_heads=4, dropout_rate=0.1):
        super().__init__()
        self.dim = dim
        self.num_gates = num_gates
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        self.query_projection = nn.Linear(dim, dim)
        self.key_projection = nn.Linear(dim, dim)
        self.value_projection = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=-1)

        # This linear layer projects the output of the attention mechanism to the gate scores
        self.gate_score_projection = nn.Linear(dim, num_gates)

    def forward(self, input1, input2, reduce_token=False):
        # Input1 and Input2 could be batched embeddings of different characteristics
        # Query from input1, Keys and Values from input2
        queries = self.query_projection(input1)  # [batch_size, seq_length, dim]
        keys = self.key_projection(input2)       # [batch_size, seq_length, dim]
        values = self.value_projection(input2)   # [batch_size, seq_length, dim]

        # Compute attention scores
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.dim ** 0.5)
        attention_scores = self.softmax(attention_scores)  # [batch_size, seq_length, seq_length]
        attention_scores = self.dropout(attention_scores)

        # Apply attention scores to the values
        attention_output = torch.matmul(attention_scores, values)  # [batch_size, seq_length, dim]

        if reduce_token:
            # Reduce the sequence dimension by averaging and remove the singleton dimension
            attention_output = attention_output.mean(dim=1)  # [batch_size, dim]

        # Generate gate scores
        gate_scores = self.gate_score_projection(attention_output)  # [batch_size, num_gates]
        gate_scores = gate_scores.softmax(dim=-1)  # Softmax over gates dimension

        return gate_scores
class Top2Gating(nn.Module):
    MIN_EXPERT_CAPACITY = 4
    
    def __init__(
        self,
        dim,
        num_gates,
        eps = 1e-9,
        outer_expert_dims = tuple(),
        second_policy_train = 'random',
        second_policy_eval = 'random',
        second_threshold_train = 0.2,
        second_threshold_eval = 0.2,
        capacity_factor_train = 1.25,
        capacity_factor_eval = 2.,
    ):
        super().__init__()

        self.eps = eps
        self.num_gates = num_gates
        self.w_gating = nn.Parameter(torch.randn(*outer_expert_dims, dim, num_gates))

        self.second_policy_train = second_policy_train
        self.second_policy_eval = second_policy_eval
        self.second_threshold_train = second_threshold_train
        self.second_threshold_eval = second_threshold_eval
        self.capacity_factor_train = capacity_factor_train
        self.capacity_factor_eval = capacity_factor_eval

    @staticmethod
    def top1(tensor):
        values, index = tensor.topk(k=1, dim=-1)
        values, index = map(lambda x: x.squeeze(dim=-1), (values, index))
        return values, index

    def forward(self, x, reduce_token=False):
        *_, b, group_size, dim = x.shape
        if reduce_token:
            group_size = 1
        num_gates = self.num_gates

        if self.training:
            policy = self.second_policy_train
            threshold = self.second_threshold_train
        else:
            policy = self.second_policy_eval
            threshold = self.second_threshold_eval

        raw_gates = torch.einsum('...bnd,...de->...bne', x, self.w_gating)
        if reduce_token:
            raw_gates = raw_gates.mean(dim=1).unsqueeze(dim=1)
        raw_gates = raw_gates.softmax(dim=-1)

        # FIND TOP 2 EXPERTS PER POSITON
        # Find the top expert for each position. shape=[batch, group]

        gate_1, index_1 = self.top1(raw_gates)
        mask_1 = F.one_hot(index_1, num_gates).float()
        
        gates_without_top_1 = raw_gates * (1. - mask_1)

        gate_2, index_2 = self.top1(gates_without_top_1)
        mask_2 = F.one_hot(index_2, num_gates).float()

        # normalize top2 gate scores
        denom = gate_1 + gate_2 + self.eps
        gate_1 /= denom
        gate_2 /= denom

        # Depending on the policy in the hparams, we may drop out some of the
        # second-place experts.
        if policy == "all":
            pass
        elif policy == "none":
            mask_2 = torch.zeros_like(mask_2)
        elif policy == "threshold":
            mask_2 *= (gate_2 > threshold).float()
        elif policy == "random":
            probs = torch.zeros_like(gate_2).uniform_(0., 1.)
            mask_2 *= (probs < (gate_2 / max(threshold, self.eps))).float().unsqueeze(-1)
        else:
            raise ValueError(f"Unknown policy {policy}")

        if reduce_token:
            soft_gate = torch.zeros(b, num_gates).to(gate_1.device)
            soft_gate = soft_gate.to(gate_1.dtype)
            soft_gate.scatter_(1, index_1, gate_1)
            soft_gate = soft_gate.to(gate_2.dtype)
            soft_gate.scatter_(1, index_2, gate_2)
        
        else:
            soft_gate = torch.zeros(b * group_size, num_gates).to(gate_1.device)
            soft_gate = soft_gate.to(gate_1.dtype)
            soft_gate.scatter_(1, index_1.view(-1, 1), gate_1.view(-1, 1))
            soft_gate = soft_gate.to(gate_2.dtype) 
            soft_gate.scatter_(1, index_2.view(-1, 1), gate_2.view(-1, 1))
            soft_gate = soft_gate.reshape(b, group_size, num_gates).contiguous()

        return soft_gate
    
# Example initialization and forward call
dim = 512
num_gates = 10
num_heads = 8
# model = CrossAttentionBasedGating(dim, num_gates, num_heads)
model = Top2Gating(dim, num_gates)
input1 = torch.randn(32, 100, dim)  # Batch size of 32, sequence length of 100
input2 = torch.randn(32, 100, dim)
soft_gate = model(input1)
# gate_scores = model(input1, input2, reduce_token=True)
print(soft_gate)  # Expected shape: [32, num_gates]
