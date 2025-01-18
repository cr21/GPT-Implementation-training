
import re

def get_raw_text(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as file:
        text= file.read()
    print(f"total number of tokens: {len(text)}")
    print("print first 100 characters: ", text[:100])

    return text


class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str_to_int 
            else "<|unk|>" for item in preprocessed
        ]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text
    

if __name__ == "__main__":
    raw_text = get_raw_text("the-verdict.txt")
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    print(f"total number of preprocessed tokens: {len(preprocessed)}")
    print(preprocessed[:30])
    all_words = sorted(set(preprocessed))
    vocab_size = len(all_words)
    print(f"total number of unique tokens: {vocab_size}")
    print(all_words[:30])
    vocab = {token:integer for integer,token in enumerate(all_words)}
    
    tokenizer = SimpleTokenizerV2(vocab)

    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the palace."

    text = " <|endoftext|> ".join((text1, text2))

    print(text)