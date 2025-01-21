import urllib.request
import zipfile
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import tiktoken
import pandas as pd
from gpt_download import download_and_load_gpt2
from gpt_model_dummy import DummyGPTModel
from gpt_pretrained_openai import load_weights_into_gpt
from gpt_model_dummy import generate_text_simple
from chapter5 import generate, text_to_token_ids, token_ids_to_text

url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "sms_spam_collection.zip"
extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

def download_and_unzip_spam_data(
    url, zip_path, extracted_path, data_file_path):
    if data_file_path.exists():
        print("Data already downloaded")
        return
    with urllib.request.urlopen(url) as response:
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File downloaded and saved as {data_file_path}")

def create_balanced_dataset(df):
    spam_df = df[df["Label"] == "spam"]
    ham_df = df[df["Label"] == "ham"]
    ham_subset = ham_df.sample(n=len(spam_df),random_state=123)
    balanced_df = pd.concat([ham_subset, spam_df])
    return balanced_df


def random_split(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    df = df.sample(
        frac=1, random_state=123
        ).reset_index(drop=True)
    train_end = int(len(df)*train_ratio)
    train_df = df[:train_end]
    val_end = train_end + int(len(df)*val_ratio)
    val_df = df[train_end:val_end]
    test_df = df[val_end:]
    return train_df, val_df, test_df




class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)
        self.encoded_text = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]
        if max_length is  None:
            self.max_length = max(len(text) for text in self.encoded_text)
        else:
            self.max_length = max_length
        self.encoded_text = [ text[:self.max_length] for text in self.encoded_text]
        self.encoded_text = [text + [pad_token_id]* (self.max_length - len(text)) for text in self.encoded_text]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        encoded = self.encoded_text[idx]
        label = self.data.iloc[idx]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )


def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0,0
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i, (inputs, labels) in enumerate(data_loader):
        if i < num_batches:
            input_batch = inputs.to(device)
            target_batch = labels.to(device)
            with torch.no_grad():
                outputs = model(input_batch)
            logits = outputs[:, -1, :] # take last token logits ( last token attention has full context)
            predicted_labels = torch.argmax(logits, dim=-1)
            num_examples += predicted_labels.shape[0]
            correct_predictions += (
                (predicted_labels == target_batch).sum().item()
            )
            
        else:
            break

    return correct_predictions / num_examples

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    outputs = model(input_batch)
    logits = outputs[:, -1, :]
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    #model.train()
    total_loss = 0.0
    num_examples = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i, (inputs, labels) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(inputs, labels, model, device)
            total_loss += loss.item()
            #num_examples += input_batch.shape[0]
        else:
            break
    return total_loss / num_batches

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def train_classifier_simple( model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            global_step += 1
            examples_seen += input_batch.shape[0]
            train_losses.append(loss.item())
            if global_step % eval_freq == 0:
                # train_acc = calc_accuracy_loader(train_loader, model, device, eval_iter)
                # val_acc = calc_accuracy_loader(val_loader, model, device, eval_iter)
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                f"Train loss {train_loss:.3f}, "
                f"Val loss {val_loss:.3f}"
                )
        train_accuracy = calc_accuracy_loader(
        train_loader, model, device, num_batches=eval_iter
        )
        val_accuracy = calc_accuracy_loader(
        val_loader, model, device, num_batches=eval_iter
        )
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)
    print("Done Training")
    return train_losses, val_losses, train_accs, val_accs, examples_seen


def plot_values(epochs_seen, examples_seen, train_values, val_values,label="loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(
    epochs_seen, val_values, linestyle="-.",
    label=f"Validation {label}"
    )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()
    ax2 = ax1.twiny()
    ax2.plot(examples_seen, train_values, alpha=0)
    ax2.set_xlabel("Examples seen")
    fig.tight_layout()
    plt.savefig(f"{label}-plot.pdf")
    plt.show()

def classify_review(model, tokenizer, text, device, max_length=None, pad_token_id=50256):
    model.eval()
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[1]
    print(f"supported_context_length {supported_context_length}")
    input_ids = input_ids[:min(supported_context_length, max_length)]
    input_ids = input_ids + [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(
        input_ids, device=device
    ).unsqueeze(0) # B,T
    with torch.no_grad():
        outputs = model(input_tensor)
    logits = outputs[:, -1, :]
    predicted_label= torch.argmax(logits, dim=-1).item()
    return  "spam" if predicted_label == 1 else "not spam"

if __name__ == "__main__":
    download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)
    import pandas as pd
    df = pd.read_csv(
    data_file_path, sep="\t", header=None, names=["Label", "Text"]
    )
    print(df.head())
    print(df["Label"].value_counts())



    balanced_df = create_balanced_dataset(df)
    print(balanced_df["Label"].value_counts())

    balanced_df['Label']=balanced_df['Label'].map({'spam':1,'ham':0})
    print(balanced_df.tail())
    train_df, val_df, test_df = random_split(balanced_df, train_ratio=0.7, val_ratio=0.1,
                                            test_ratio=0.2)
    print(train_df.shape, val_df.shape, test_df.shape)

    train_df.to_csv("train.csv", index=None)
    val_df.to_csv("validation.csv", index=None)
    test_df.to_csv("test.csv", index=None)

    tokenizer = tiktoken.get_encoding("gpt2")
    train_dataset = SpamDataset(
        csv_file="train.csv",
        max_length=None,
        tokenizer=tokenizer
    )
    val_dataset = SpamDataset(
        csv_file="validation.csv",
        max_length=None,
        tokenizer=tokenizer
    )
    test_dataset = SpamDataset(
        csv_file="test.csv",
        max_length=None,
        tokenizer=tokenizer
    )   
    num_workers = 0
    batch_size = 8
    torch.manual_seed(123)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

    for input_batch, target_batch in train_loader:
        pass
    print("Input batch dimensions:", input_batch.shape)
    print("Label batch dimensions", target_batch.shape)

    print(f"{len(train_loader)} training batches")
    print(f"{len(val_loader)} validation batches")
    print(f"{len(test_loader)} test batches")

    ## Model pretrained weight initialization
    CHOOSE_MODEL = "gpt2-small (124M)"
    INPUT_PROMPT = "Every effort moves"
    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True
    }
    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
    

    
    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    print(f"Model size: {model_size}")
    settings, params = download_and_load_gpt2(
        model_size=model_size, models_dir="gpt2"
    )
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = DummyGPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model = model.to(device)
    model.eval()

    text_1 = "Every effort moves you"
    token_ids = generate_text_simple(
    model=model,
    tokenizer=tokenizer,
    idx=text_to_token_ids(text_1, tokenizer),
    max_new_tokens=15,
    context_length=BASE_CONFIG["context_length"]
    )
    print(token_ids_to_text(token_ids, tokenizer))

    text_2 = (
        "Is the following text 'spam'? Answer with 'yes' or 'no':"
        " 'You are a winner you have been specially"
        " selected to receive $1000 cash or a $2000 award.'"
    )
    token_ids = generate_text_simple(
    model=model,
    tokenizer=tokenizer,
    idx=text_to_token_ids(text_2, tokenizer),
    max_new_tokens=23,
    context_length=BASE_CONFIG["context_length"]
    )
    print(token_ids_to_text(token_ids, tokenizer))
    print(model)

    for param in model.parameters():
        param.requires_grad = False

    print(model)
    torch.manual_seed(123)
    num_classes = 2
    model.lm_head = torch.nn.Linear(
        in_features=BASE_CONFIG["emb_dim"],
        out_features=num_classes
    )
    # train final layer norm and last transformer block trainable
    for param in model.ln_final.parameters():
        param.requires_grad = True
    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True
    
    inputs = tokenizer.encode("Do you have time")
    inputs = torch.tensor(inputs).unsqueeze(0) # add batch dimension [B, T]
    print("Inputs:", inputs)
    print("Inputs dimensions:", inputs.shape)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    with torch.no_grad():
        outputs = model(inputs.to(device))
    print("Outputs:\n", outputs)
    print("Outputs dimensions:", outputs.shape) # [B, T, 2]
    print("Last output token:", outputs[:, -1, :])

    probas = torch.softmax(outputs[:, -1, :], dim=-1)
    label = torch.argmax(probas)
    print("Class label:", label.item())


    ## accuracy test
    model.to(device)
    torch.manual_seed(123)
    import time
    start_time = time.time()
    torch.manual_seed(123)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    num_epochs = 5
    train_losses, val_losses, train_accs, val_accs, examples_seen = \
        train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=50,
        eval_iter=5
    )
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    



    # accuracy test
    train_accuracy = calc_accuracy_loader(
    train_loader, model, device, num_batches=10
    )
    val_accuracy = calc_accuracy_loader(
    val_loader, model, device, num_batches=10
    )
    test_accuracy = calc_accuracy_loader(
    test_loader, model, device, num_batches=10
    )
    print(f"Training accuracy: {train_accuracy*100:.2f}%")
    print(f"Validation accuracy: {val_accuracy*100:.2f}%")
    print(f"Test accuracy: {test_accuracy*100:.2f}%")
    # loss test
    with torch.no_grad():
        print("Calculating loss")
    train_loss = calc_loss_loader(
        train_loader, model, device, num_batches=5
    )
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
    test_loss = calc_loss_loader(test_loader, model, device, num_batches=5)
    print(f"Training loss: {train_loss:.3f}")
    print(f"Validation loss: {val_loss:.3f}")
    print(f"Test loss: {test_loss:.3f}")  


    # classify review
    text_1 = (
    "You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award."
    )
    print(classify_review(
    model, tokenizer, text_1, device, max_length=train_dataset.max_length
    ))

    text_2 = (
    "Hey, just wanted to check if we're still on"
    " for dinner tonight? Let me know!"
    )
    print(classify_review(
    model, tokenizer, text_2, device, max_length=train_dataset.max_length
    ))

    torch.save(model.state_dict(), "review_classifier.pth")
    model_state_dict = torch.load("review_classifier.pth", map_location=device)
    model.load_state_dict(model_state_dict)

