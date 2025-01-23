from gpt_download import download_and_load_gpt2
from gpt_model_dummy import DummyGPTModel
from gpt_pretrained_openai import load_weights_into_gpt
from gpt_model_dummy import generate_text_simple
from chapter5 import generate, text_to_token_ids, token_ids_to_text, calc_loss_loader, train_simple_model, plot_losses
from chapter6 import generate_text_simple, generate
from chapter7 import download_and_load_file, InstructionDataset, format_input, final_collate_fn, customized_collate_fn
import torch
from torch.utils.data import DataLoader
import tiktoken
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
BASE_CONFIG = {
    "vocab_size": 50257, # Vocabulary size
    "context_length": 1024, # Context length
    "drop_rate": 0.0, # Dropout rate
    "qkv_bias": True # Query-key-value bias
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

if __name__ == "__main__":
    CHOOSE_MODEL = "gpt2-small (124M)"
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(
    model_size=model_size,
    models_dir="gpt2"
    )
    torch.manual_seed(123)
    

    # load data
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
    input_text = format_input(val_data[0])
    model = DummyGPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    #model.to(device)
    #model.eval();

    # create dataloader
    num_workers = 0
    batch_size = 1
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
    
    input_text = format_input(val_data[0])
    print(input_text)
    # token_ids = generate(
    #     model=model, 
    #     idx=text_to_token_ids(input_text, tokenizer),  
    #     max_new_tokens=35,
    #     context_length=BASE_CONFIG["context_length"],
    #     device=device, 
    #     eos_token=50256,
    #     top_k=1
    # )
    # print("--------------------------------")
    # generated_text = token_ids_to_text(token_ids, tokenizer)
    # print("generated text:")
    # print(generated_text)

    # Before Training Loss
    # model.to(device)
    # torch.manual_seed(123)
    # with torch.no_grad():
    #     train_loss = calc_loss_loader(
    #     train_loader, model, device, num_batches=5
    #     )
    #     val_loss = calc_loss_loader(
    #     validation_loader, model, device, num_batches=5
    #     )
    # print("Training loss:", train_loss)
    # print("Validation loss:", val_loss)
    # torch.save(model.state_dict(), "instruction_model.pth")
    # model_state_dict = torch.load("instruction_model.pth", map_location=device)
    # model.load_state_dict(model_state_dict)

    # train model
    #train_simple_model(model, train_loader, validation_loader, device, num_epochs=10)
    import time
    start_time = time.time()
    torch.manual_seed(123)
    import gc
    gc.collect()
    if device.type == 'mps':
        torch.mps.empty_cache()
    
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
    model.parameters(), lr=0.00005, weight_decay=0.1
    )
    num_epochs = 3
    train_losses, val_losses, tokens_seen = train_simple_model(
    model, tokenizer, train_loader, validation_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context=format_input(val_data[0])
    )
    gc.collect()
    if device.type == 'mps':
        torch.mps.empty_cache()
    end_time = time.time()
    torch.save(model.state_dict(), "instruction_model.pth")
    print(f"Training time: {end_time - start_time:.2f} seconds") 
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))   
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)