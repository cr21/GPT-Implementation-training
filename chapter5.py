from gpt_model_dummy import DummyGPTModel, generate_text_simple
import torch
import tiktoken
from gptdataloader import create_dataloader_v1
from plot_model import plot_losses
torch.manual_seed(123)
GPT_CONFIG_124M = {
    "vocab_size": 50257, # Vocabulary size
    "context_length": 256, # Context length
    "emb_dim": 768, # Embedding dimension
    "n_heads": 12, # Number of attention heads
    "n_layers": 12, # Number of layers
    "drop_rate": 0.1, # Dropout rate
    "qkv_bias": False # Query-Key-Value bias
}


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    return torch.tensor(encoded).unsqueeze(0)

def token_ids_to_text(token_ids, tokenizer):
    decoded = tokenizer.decode(token_ids.squeeze(0).tolist())
    return decoded

def calculate_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
    logits.flatten(0, 1), target_batch.flatten()
    )
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calculate_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def evaluate_model(model, train_dataloader, valid_dataloader, device, eval_iter=100):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_dataloader, model, device, num_batches=eval_iter)
        valid_loss = calc_loss_loader(valid_dataloader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, valid_loss

def generate_and_print_text(model, tokenizer, start_context, device):
    model.eval()
    print(model.pos_emb.weight.shape)
    context_size = model.pos_emb.weight.shape[0]
    token_ids = generate_text_simple(model=model,
                                 tokenizer=tokenizer,
                                 idx=text_to_token_ids(start_context, tokenizer),
                                 max_new_tokens=50,
                                 context_length=context_size)
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()

def train_simple_model(model, tokenizer, train_dataloader, valid_dataloader,
                        optimizer, device, num_epochs, eval_freq, eval_iter, start_context):
    train_losses, valid_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_dataloader:
            optimizer.zero_grad()
            loss = calculate_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1
            if global_step % eval_freq == 0:
                train_loss, valid_loss = evaluate_model(model, train_dataloader, valid_dataloader, device)
                train_losses.append(train_loss)
                valid_losses.append(valid_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, "
                    f"Val loss {valid_loss:.3f}"
                    )
        # after each epoch, generate text
        generate_and_print_text(model, tokenizer, start_context, device)
    return train_losses, valid_losses, track_tokens_seen

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

start_context = "Every Effort Counts"
tokenizer = tiktoken.get_encoding("gpt2")
model = DummyGPTModel(config=GPT_CONFIG_124M)
model = model.to(device)  # Move model to device

token_ids = generate_text_simple(model=model,
                               tokenizer=tokenizer,
                               idx=text_to_token_ids(start_context, tokenizer),
                               max_new_tokens=10,
                               context_length=GPT_CONFIG_124M["context_length"])
print(token_ids)
print(token_ids_to_text(token_ids, tokenizer))

### loss

inputs = torch.tensor([[16833, 3626, 6100], # ["every effort moves",
                        [40, 1107, 588]]) # "I really like"]
targets = torch.tensor([[3626, 6100, 345], # ["effort moves you",
                        [1107, 588, 11311]]) # "really like chocolate"]
print(inputs.shape)
print(targets.shape)
with torch.no_grad():
    logits = model(inputs)
logits =torch.softmax(logits, dim=-1)
print(logits.shape)
token_ids = torch.argmax(logits, dim=-1, keepdim=True)
print(token_ids.shape)
print(token_ids)
print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
print(f"Outputs batch 1:"
f" {token_ids[0].flatten()} {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")
text_idx = 0
target_probs1=logits[text_idx, [0, 1, 2], targets[text_idx]]
print(f"Target probs batch 1: {target_probs1}")
print(f"Target probs batch 1: {target_probs1}")
text_idx = 1
print(logits.shape)
print(targets[text_idx])
target_probas_2 = logits[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 2:", target_probas_2)




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
model = DummyGPTModel(config=GPT_CONFIG_124M)
model.to(device)
with torch.no_grad():
    train_loss = calc_loss_loader(train_dataloader, model, device)
    valid_loss = calc_loss_loader(valid_dataloader, model, device)
print(f"Train loss: {train_loss}")
print(f"Validation loss: {valid_loss}")

# train model
torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0004, weight_decay=0.1
)
num_epochs = 10
train_losses, val_losses, tokens_seen = train_simple_model(
    model, tokenizer, train_dataloader, valid_dataloader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="Every effort moves you"
)
print(train_losses)
print(val_losses)
print(tokens_seen)
epochs_tensor = torch.linspace(0, 10, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
