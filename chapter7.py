# download the instruction dataset
import json
import os
from urllib.request import urlopen
from torch.utils.data import Dataset
import torch
import tiktoken
from functools import partial
from torch.utils.data import DataLoader

def download_and_load_file(file_path, url):
    if not os.path.exists(file_path):
        with urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = (
        f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    )
    return instruction_text + input_text

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            output_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + output_text
            self.encoded_texts.append(tokenizer.encode(full_text))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.encoded_texts[idx]
    
def custom_collate_fn_draft1(batch, padding_token_id, device="cpu"):
    batch_max_length = max(len(text)+1 for text in batch)
    input_lsts =[]
    for item in batch:
        new_item = item.copy()
        new_item += [padding_token_id]
    
        padded  = (
            new_item + [padding_token_id] * (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])
        input_lsts.append(inputs)
    inputs_tensor = torch.stack(input_lsts).to(device)
    return inputs_tensor


def custom_collate_fn_draft2(batch, padding_token_id, device="cpu"):
    batch_max_length = max(len(text)+1 for text in batch)
    input_lsts = []
    target_lsts = []
    for item in batch:
        new_item = item.copy()
        new_item += [padding_token_id]
        padded = new_item + [padding_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])
        input_lsts.append(inputs)
        target_lsts.append(targets)
    inputs_tensor = torch.stack(input_lsts).to(device)
    targets_tensor = torch.stack(target_lsts).to(device)
    return inputs_tensor, targets_tensor


def final_collate_fn(batch, 
                     padding_token_id=50256, 
                     device="cpu", 
                     ignore_index=-100,
                       allowed_max_length=None):
    batch_max_length = max(len(text)+1 for text in batch)
    inputs_lsts , targets_lsts = [], []
    for item in batch:
        new_item = item.copy()
        new_item += [padding_token_id]
        padded = new_item + [padding_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])
        # remove padding tokens from targets
        
        mask = targets == padding_token_id
        indices = torch.nonzero(mask)
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index
        # truncate to allowed max length
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]
        inputs_lsts.append(inputs)
        targets_lsts.append(targets)
    inputs_tensor = torch.stack(inputs_lsts).to(device)
    targets_tensor = torch.stack(targets_lsts).to(device)
    return inputs_tensor, targets_tensor


device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
customized_collate_fn = partial(
    final_collate_fn,
    device=device,
    allowed_max_length=1024
    )
if __name__ == "__main__":
    file_path = "instruction-data.json"
    url = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
    "/main/ch07/01_main-chapter-code/instruction-data.json"
    )
    data = download_and_load_file(file_path, url)
    print("Number of entries:", len(data))
    print("Example entry:\n", data[50])
    print("Another example entry:\n", data[999])
    model_input = format_input(data[50])
    desired_response = f"\n\n### Response:\n{data[50]['output']}"
    print(model_input + desired_response)

    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.1)
    val_portion = len(data) - train_portion - test_portion

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]
    print("Number of training samples:", len(train_data))
    print("Number of test samples:", len(test_data))
    print("Number of validation samples:", len(val_data))

    # dataset test
    tokenizer = tiktoken.get_encoding("gpt2")
    print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))


    # test custom collate fn
    inputs = [
        [0,1,2,3,4,5],
        [0,1,2],
        [2,3,4,5]
    ]
    print(custom_collate_fn_draft1(inputs, padding_token_id=50256))

    # test custom collate fn draft 2
    inputs, targets = custom_collate_fn_draft2(inputs, padding_token_id=50256)

    print(f"Input ")
    print(inputs)
    print("+"*50)
    print("Targets")
    print(targets)

    # test final collate fn
    inputs_1 = [0, 1, 2, 3, 4]
    inputs_2 = [5, 6]
    inputs_3 = [7, 8, 9]
    batch = (
        inputs_1,
        inputs_2,
        inputs_3
    )
    print("*"*50)
    inputs, targets = final_collate_fn(batch, padding_token_id=50256)
    
    print(f"Input ")
    print(inputs)
    print("+"*50)
    print("Targets")
    print(targets)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # datalaoder

    num_workers = 0
    batch_size = 8
    torch.manual_seed(123)
    train_dataset = InstructionDataset(train_data, tokenizer)

    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size,
                                num_workers=num_workers, 
                                collate_fn=customized_collate_fn,
                                shuffle=True,
                                drop_last=True)
    validation_dataset = InstructionDataset(val_data, tokenizer)
    validation_loader = DataLoader(validation_dataset,
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    collate_fn=customized_collate_fn,
                                    shuffle=False,
                                    drop_last=False)

    test_dataset = InstructionDataset(test_data, tokenizer)
    test_loader = DataLoader(test_dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            collate_fn=customized_collate_fn,
                            shuffle=False,
                            drop_last=False)
    
    print("Train loader:")
    for inputs, targets in train_loader:
        print(inputs.shape, targets.shape)

    print("Validation loader:")
    for inputs, targets in validation_loader:
        print(inputs.shape, targets.shape)

    print("Test loader:")
    for inputs, targets in test_loader:
        print(inputs.shape, targets.shape)