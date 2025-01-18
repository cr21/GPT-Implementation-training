import torch
from torch.utils.data import Dataset, DataLoader

class GPTDataLoader(Dataset):
    def __init__(self, txt:str, tokenizer, context_size:int, stride:int):
        self.input_ids =[]
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - context_size, stride):
            input_chunk = token_ids[i:i + context_size]
            target_chunk = token_ids[i + 1: i + context_size + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
            
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]