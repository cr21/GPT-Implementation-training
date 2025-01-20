import torch
from torch import nn
import tiktoken
import matplotlib.pyplot as plt
from multihead_attention_efficient import MultiHeadAttentionEfficient

GPT_CONFIG_124M = {
    "vocab_size": 50257, # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768, # Embedding dimension
    "n_heads": 12, # Number of attention heads
    "n_layers": 12, # Number of layers
    "drop_rate": 0.1, # Dropout rate
    "qkv_bias": False # Query-Key-Value bias
}


class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) *
                                          (x + 0.044715 * torch.pow(x, 3))
                            ))

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )
    def forward(self, x):
        return self.layers(x)

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attn = MultiHeadAttentionEfficient(d_in=config["emb_dim"],
                                                d_out=config["emb_dim"],
                                               context_length=config["context_length"],
                                               dropout=config["drop_rate"],
                                               num_heads=config["n_heads"],
                                               qkv_bias=config["qkv_bias"])
        self.ff = FeedForward(config)
        self.norm1 = LayerNorm(config["emb_dim"])
        self.norm2 = LayerNorm(config["emb_dim"])
        self.drop_shortcut = nn.Dropout(config["drop_rate"])
    
    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.drop_shortcut(x)
        x = shortcut + x

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = shortcut + x
        return x

class DummyGPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.pos_emb = nn.Embedding(config["context_length"], config["emb_dim"])
        self.drop_emb = nn.Dropout(config["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(config)
            for _ in range(config["n_layers"])]
        )
        self.ln_final = LayerNorm(config["emb_dim"])
        self.lm_head = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)
    
    def forward(self, x):
        batch_size, seq_len = x.shape
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(torch.arange(0, seq_len, device=x.device))
        x = tok_emb + pos_emb
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.ln_final(x)
        output = self.lm_head(x)
        return output
    

def generate_text_simple(model, tokenizer, idx, max_new_tokens, context_length):
    # idx (batch_size, n_tokens)
    for _ in range(max_new_tokens):
        # take maximum last 5 of context length only 
        idx_cond = idx[:, -context_length:] # (batch_size, context_length)
        with torch.no_grad():
            # idx (batch_size, n_tokens + 1)
            logits = model(idx_cond) # batchsize, n_tokens, vocab_size
        # take last token logits 
        logits = logits[:, -1, :]  # batchsize, vocab_size
        probas = torch.softmax(logits, dim=-1) # batchsize, vocab_size
        idx_next = torch.argmax(probas, dim=-1, keepdim=True) # batchsize, 1
        idx = torch.cat((idx, idx_next), dim=1) # batchsize, n_tokens + 1
        
    return idx

if __name__ == "__main__":
    print(GPT_CONFIG_124M)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    tokenizer = tiktoken.get_encoding("gpt2")
    batch =[]
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"
    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0)
    # print(batch.shape)
    # print(batch)
    # torch.manual_seed(123)
    # model = DummyGPTModel(GPT_CONFIG_124M)
    # logits = model(batch)
    # print("Output shape:", logits.shape)
    # print(logits.shape)
    
    # layernorm test
    torch.manual_seed(123)
    batch_example = torch.randn(2, 5)
    ln = LayerNorm(emb_dim=5)
    out_ln = ln(batch_example)
    mean = out_ln.mean(dim=-1, keepdim=True)
    var = out_ln.var(dim=-1, keepdim=True, unbiased=False)
    print(mean)
    print(var)
    
    
    # feedforward test
    torch.manual_seed(123)
    batch_example = torch.randn(2, 3, GPT_CONFIG_124M["emb_dim"])
    ff = FeedForward(cfg=GPT_CONFIG_124M)
    out_ff = ff(batch_example)
    print(out_ff.shape)

    # transformer block test
    torch.manual_seed(123)
    batch_example = torch.rand(2, 4, 768)
    trf_block = TransformerBlock(config=GPT_CONFIG_124M)
    out_trf_block = trf_block(batch_example)
    print(out_trf_block.shape)
    print("Input shape:", batch_example.shape)
    print("Output shape:", out_trf_block.shape)


    # gpt model test
    torch.manual_seed(123)
    model = DummyGPTModel(config=GPT_CONFIG_124M)
    out_model = model(batch)
    print("Input batch:\n", batch)
    print(out_model)    
    print("\nOutput shape:", out_model.shape)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    print("Token embedding layer shape:", model.tok_emb.weight.shape)
    print("Output layer shape:", model.lm_head.weight.shape)
    total_params_gpt2 = (
    total_params - sum(p.numel()
    for p in model.lm_head.parameters())
    )
    print(f"Number of trainable parameters "
    f"considering weight tying: {total_params_gpt2:,}"
    )
    
    total_size_bytes = total_params * 4
    total_size_mb = total_size_bytes / (1024 * 1024)
    print(f"Total size of the model: {total_size_mb:.2f} MB")

    # generate text test
    start_context = "Hello, I am"
    encoded = tokenizer.encode(start_context)
    print(f"Encoded: {encoded}")
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    print("encoded_tensor.shape:", encoded_tensor.shape)# (1, num_tokens)
    model.eval()
    out_generate_text = generate_text_simple(model=model,
                                              tokenizer=tokenizer,
                                                idx=encoded_tensor,
                                                  max_new_tokens=6,
                                                    context_length=GPT_CONFIG_124M["context_length"])
    print(out_generate_text)
    print("Output length:", len(out_generate_text[0]))
    decoded_text = tokenizer.decode(out_generate_text.squeeze(0).tolist())
    print(decoded_text)
