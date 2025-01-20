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
    
from bytepair_encoding import BPETokenizer
from gptdataloader import GPTDataLoader
from torch.utils.data import DataLoader
from torch import nn
import torch
def create_dataloader_v1(txt:str, batch_size:int=4, context_size:int=256, stride:int=128,
                       tokenizer:BPETokenizer=None, shuffle:bool=True, drop_last:bool=True,
                       num_workers:int=2):
    bpe_tokenizer = BPETokenizer("gpt2")

    dataset = GPTDataLoader(txt, bpe_tokenizer, 
                            context_size=context_size, 
                            stride=stride)
    gpt_dataloader = DataLoader(dataset, 
                                batch_size=batch_size,
                                shuffle=shuffle,
                                drop_last=drop_last,
                                num_workers=num_workers)
    return gpt_dataloader