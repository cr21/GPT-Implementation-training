import torch
import torch.nn as nn


class MultiHeadAttentionEfficient(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
            diagonal=1)
        )

    def forward(self, x):
        batch_size, n_tokens, d_in = x.shape
        keys = self.W_key(x) # B,T,D
        queries = self.W_query(x) # (batch_size, n_tokens, d_out)
        values = self.W_value(x) # (batch_size, n_tokens, d_out)


        keys = keys.view(batch_size, n_tokens, self.num_heads, self.head_dim) # B, T, H, D
        queries = queries.view(batch_size, n_tokens, self.num_heads, self.head_dim) # B, T, H, D
        values = values.view(batch_size, n_tokens, self.num_heads, self.head_dim) # B, T, H, D


        keys = keys.transpose(1, 2) # B, H, T, D
        queries = queries.transpose(1, 2) # B, H, T, D
        values = values.transpose(1, 2) # B, H, T, D

        # Compute attention scores
        attn_scores = queries @ keys.transpose(-2, -1) # B, H, T, D * B, H,D,T =  B, H, T, T
        mask_bool = self.mask.bool()[:n_tokens, :n_tokens] # T, T
        attn_scores.masked_fill_(mask_bool, -torch.inf) # B, H, T, T
        attn_weights = torch.softmax(attn_scores/keys.shape[-1]**0.5, dim=-1) # B, H, T, T
        attn_weights = self.dropout(attn_weights)
        context_vector = attn_weights @ values # B, H, T, T * B, H, T, D = B, H, T, D
        context_vector = context_vector.transpose(1, 2) # B, T, H, D
        context_vector = context_vector.contiguous().view(batch_size, n_tokens, self.d_out) # B, T, H, D -> B, T, D
        return self.out_proj(context_vector)


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
    mha_eff = MultiHeadAttentionEfficient(d_in, d_out, context_length, 0.0, 2)
    #causal_attention = CausalAttention(d_in, d_out, context_length, 0.0)
    mha_eff_vector = mha_eff(batch)
    print("mha_eff_vector.shape:", mha_eff_vector.shape)
    print(mha_eff_vector)
    
