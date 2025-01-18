from importlib.metadata import version
import tiktoken


class BPETokenizer:
    def __init__(self, model_name: str):
        self.tokenizer = tiktoken.get_encoding(model_name)

    def encode(self, text: str):
        return self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    
    def decode(self, ids: list[int]):
        return self.tokenizer.decode(ids)

if __name__ == "__main__":
    tokenizer = BPETokenizer("gpt2")
    text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace."
    integers = tokenizer.encode(text)
    print(integers)
    print(tokenizer.decode(integers))

