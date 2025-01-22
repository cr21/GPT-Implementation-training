from gpt_model_dummy import DummyGPTModel
import torch
import tiktoken
from gpt_download import download_and_load_gpt2
from chapter5 import generate, text_to_token_ids, token_ids_to_text
from chapter5 import calc_loss_loader
from gptdataloader import create_dataloader_v1
model_configs = {
"gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
"gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
"gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
"gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

GPT_CONFIG_124M = {
    "vocab_size": 50257, # Vocabulary size
    "context_length": 256, # Context length
    "emb_dim": 768, # Embedding dimension
    "n_heads": 12, # Number of attention heads
    "n_layers": 12, # Number of layers
    "drop_rate": 0.1, # Dropout rate
    "qkv_bias": False # Query-Key-Value bias
}
model_name = "gpt2-small (124M)"
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])
NEW_CONFIG.update({"context_length": 1024})
NEW_CONFIG.update({"qkv_bias": True})



# for name, param in gpt.named_parameters():
#     print(name, param.shape)

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, "
        "Right: {right.shape}"
        )
    return torch.nn.Parameter(torch.tensor(right))

import numpy as np

def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])
    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
        (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].attn.W_query.weight = assign(
        gpt.trf_blocks[b].attn.W_query.weight, q_w.T)
        gpt.trf_blocks[b].attn.W_key.weight = assign(
        gpt.trf_blocks[b].attn.W_key.weight, k_w.T)
        gpt.trf_blocks[b].attn.W_value.weight = assign(
        gpt.trf_blocks[b].attn.W_value.weight, v_w.T)
        
        q_b, k_b, v_b = np.split(
        (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].attn.W_query.bias = assign(
        gpt.trf_blocks[b].attn.W_query.bias, q_b)
        gpt.trf_blocks[b].attn.W_key.bias = assign(
        gpt.trf_blocks[b].attn.W_key.bias, k_b)
        gpt.trf_blocks[b].attn.W_value.bias = assign(
        gpt.trf_blocks[b].attn.W_value.bias, v_b)
        gpt.trf_blocks[b].attn.out_proj.weight = assign(
        gpt.trf_blocks[b].attn.out_proj.weight,
        params["blocks"][b]["attn"]["c_proj"]["w"].T)

        gpt.trf_blocks[b].attn.out_proj.bias = assign(
        gpt.trf_blocks[b].attn.out_proj.bias,
        params["blocks"][b]["attn"]["c_proj"]["b"])
        gpt.trf_blocks[b].ff.layers[0].weight = assign(
        gpt.trf_blocks[b].ff.layers[0].weight,
        params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
        gpt.trf_blocks[b].ff.layers[0].bias,
        params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
        gpt.trf_blocks[b].ff.layers[2].weight,
        params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
        gpt.trf_blocks[b].ff.layers[2].bias,
        params["blocks"][b]["mlp"]["c_proj"]["b"])
        gpt.trf_blocks[b].norm1.scale = assign(
        gpt.trf_blocks[b].norm1.scale,
        params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
        gpt.trf_blocks[b].norm1.shift,
        params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
        gpt.trf_blocks[b].norm2.scale,
        params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
        gpt.trf_blocks[b].norm2.shift,
        params["blocks"][b]["ln_2"]["b"])

        gpt.ln_final.scale = assign(gpt.ln_final.scale, params["g"])
        gpt.ln_final.shift = assign(gpt.ln_final.shift, params["b"])
        gpt.lm_head.weight = assign(gpt.lm_head.weight, params["wte"])


if __name__ == "__main__":

    settings, params = download_and_load_gpt2(
    model_size="124M", models_dir="gpt2"
    )
    gpt = DummyGPTModel(config=NEW_CONFIG)
    gpt.eval()
    params = load_weights_into_gpt(gpt, params)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    gpt.to(device)

    print("done")

    tokenizer = tiktoken.get_encoding("gpt2")
    torch.manual_seed(123)
    token_ids = generate(
    model=gpt,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
    max_new_tokens=25,
    context_length=NEW_CONFIG["context_length"],
    top_k=50,
    temperature=1.5
    )
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))


    # exercise 5.4 


    ## calculate loss
    file_path = "the-verdict.txt"
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

    total_characters = len(text_data)
    total_tokens = len(tokenizer.encode(text_data))
    print("Characters:", total_characters)
    print("Tokens:", total_tokens)

    # create dataloader
    train_ratio = 0.9
    val_ratio = 0.1
    split_idx  =   int(train_ratio * total_characters)

    train_data = text_data[:split_idx]

    valid_data = text_data[split_idx:]

    train_dataloader = create_dataloader_v1(
        txt=train_data,
        tokenizer=tokenizer,
        batch_size=2,
        context_size=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        shuffle=True,
        drop_last=True,
        num_workers=0
    )
    valid_dataloader = create_dataloader_v1(
        txt=valid_data,
        tokenizer=tokenizer,
        batch_size=2,
        context_size=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        shuffle=False,
        drop_last=False,
        num_workers=0
    )
    print(f"Train dataset size: {len(train_dataloader)}")
    print(f"Validation dataset size: {len(valid_dataloader)}")
    print("Train loader:")
    for x, y in train_dataloader:
        print(x.shape, y.shape)
    print("\nValidation loader:")
    for x, y in valid_dataloader:
        print(x.shape, y.shape)

    # calculate loss of batch
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    with torch.no_grad():
        train_loss = calc_loss_loader(train_dataloader, gpt, device)
        valid_loss = calc_loss_loader(valid_dataloader, gpt, device)
    print(f"Train loss: {train_loss}")
    print(f"Validation loss: {valid_loss}")
