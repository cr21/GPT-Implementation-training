import torch
import torch.nn as nn

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out  ):
        super(SelfAttention_v1, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.W_query = nn.Parameter(torch.randn(d_in, d_out), requires_grad=True)
        self.W_key = nn.Parameter(torch.randn(d_in, d_out), requires_grad=True)
        self.W_value = nn.Parameter(torch.randn(d_in, d_out), requires_grad=True)

    def forward(self, x):
        # x is the input to the self-attention layer
        # x is of shape (batch_size, sequence_length, d_in)
        # we want to compute the self-attention scores for each position in the sequence
        # we do this by computing the dot product of the query and key vectors for each position
        # we then apply a softmax function to the scores to get the attention weights
        # we then use the attention weights to compute the weighted sum of the value vectors
        # the output is of shape (batch_size, sequence_length, d_out)
        keys = x @ self.W_key 
        
        query = x @ self.W_query    
        values = x @ self.W_value
        print(keys.shape, x.shape, self.W_key.shape, values.shape, query.shape)
        attn_scores = keys @ query.T
        attn_weights = torch.softmax(attn_scores/keys.shape[-1]**0.5, dim=-1)
        context_vector = attn_weights @ values
        return context_vector
    
class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super(SelfAttention_v2, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        # x is the input to the self-attention layer
        # x is of shape (batch_size, sequence_length, d_in)
        # we want to compute the self-attention scores for each position in the sequence
        # we do this by computing the dot product of the query and key vectors for each position
        # we then apply a softmax function to the scores to get the attention weights
        # we then use the attention weights to compute the weighted sum of the value vectors
        # the output is of shape (batch_size, sequence_length, d_out)
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        print(keys.shape, x.shape, queries.shape, values.shape, queries.shape)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores/keys.shape[-1]**0.5, dim=-1)
        context_vector = attn_weights @ values
        return context_vector

    
if __name__ == "__main__":
    import torch
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
    self_attention = SelfAttention_v1(d_in, d_out)
    print(self_attention(inputs))
    print(self_attention(inputs).shape)
    print("Checking v2")
    torch.manual_seed(789)
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
    self_attention_v2 = SelfAttention_v2(d_in, d_out)
    print(self_attention_v2(inputs))
    print(self_attention_v2(inputs).shape)
    
    