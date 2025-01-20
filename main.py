

from bytepair_encoding import BPETokenizer
from gptdataloader import GPTDataLoader
from torch.utils.data import DataLoader
from torch import nn
import torch

# def create_dataloader_v1(txt:str, batch_size:int=4, context_size:int=256, stride:int=128,
#                        tokenizer:BPETokenizer=None, shuffle:bool=True, drop_last:bool=True,
#                        num_workers:int=2):
#     bpe_tokenizer = BPETokenizer("gpt2")

#     dataset = GPTDataLoader(txt, bpe_tokenizer, 
#                             context_size=context_size, 
#                             stride=stride)
#     gpt_dataloader = DataLoader(dataset, 
#                                 batch_size=batch_size,
#                                 shuffle=shuffle,
#                                 drop_last=drop_last,
#                                 num_workers=num_workers)
#     return gpt_dataloader

if __name__ == "__main__":
    with open("the-verdict.txt", "r", encoding="utf-8") as file:
        raw_text = file.read()

    
    
    vocab_size = 50257
    output_dim = 768
    token_embedding = nn.Embedding(vocab_size, output_dim)
    dataloader = GPTDataLoader(raw_text, batch_size=8, 
                                   context_size=4,
                                     stride=4, 
                                     shuffle=False)

    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Token IDs:\n", inputs)
    print("\nInputs shape:\n", inputs.shape)
    print("\nTargets shape:\n", targets.shape)

    token_embeddings = token_embedding(inputs)
    print(token_embeddings.shape)
    context_size  = 4
    context_length = context_size
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))
    # print(pos_embeddings.shape)
    # print(pos_embeddings.shape)
    # print(pos_embeddings)   
    input_embeddings = token_embeddings + pos_embeddings
    print(input_embeddings.shape)