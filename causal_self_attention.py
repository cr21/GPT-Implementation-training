import torch
import torch.nn as nn

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout_rate, qkv_bias=False):
        super(CausalAttention, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.context_length = context_length
        self.dropout_rate = dropout_rate
        self.qkv_bias = qkv_bias
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_out = nn.Linear(d_out, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout_rate)
        self.register_buffer(
                'mask',
                torch.triu(torch.ones(context_length, context_length),
                diagonal=1)
        )

    def forward(self, x):
        batch_size, n_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = keys @ queries.transpose(1,2)
        # apply mask
        attn_scores.masked_fill_(
            self.mask.bool()[:n_tokens, :n_tokens], -torch.inf
        )
        attn_weights = torch.softmax(attn_scores/keys.shape[-1]**0.5, dim=-1)
        # apply dropout
        attn_weights = self.dropout(attn_weights)
        context_vector = attn_weights @ values
        return context_vector

if __name__ == "__main__":
    torch.manual_seed(123)
    inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your (x^1)
    [0.55, 0.87, 0.66], # journey (x^2)
    [0.57, 0.85, 0.64], # starts (x^3)
    [0.22, 0.58, 0.33], # with (x^4)
    [0.77, 0.25, 0.10], # one (x^5)
    [0.05, 0.80, 0.55]] # step (x^6)
    )
    d_in = inputs.shape[-1]
    d_out = 2
    batch = torch.stack((inputs, inputs), dim=0)
    batch_size = batch.shape[0]
    print(batch.shape)
    print(batch)
    torch.manual_seed(123)
    context_length = batch.shape[1]
    causal_attention = CausalAttention(d_in, d_out, context_length, 0.0)
    context_vector = causal_attention(batch)
    print("context_vecs.shape:", context_vector.shape)
    print(context_vector)
    
