from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from custom_tokenizers.bpe_tokenizer import BPETokenizer 
import torch

class WikiDataset(Dataset):
    def __init__(self, tokenizer, block_size=128, split='train', train = False):
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        text = "\n\n".join(dataset["text"])

        if  train:
            tokenizer.train(corpus_text=text)  # Train tokenizer if not already trained
        # tokens = tokenizer(text, return_tensors='pt', truncation=False)["input_ids"][0]
        tokens = tokenizer.encode(text, add_special_tokens=False)

        self.inputs = []
        # Adjust loop to avoid indexing out of bounds
        for i in range(0, len(tokens) - block_size, block_size):
             self.inputs.append(tokens[i:i+block_size+1]) # +1 for target

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        data = self.inputs[idx]
        input_ids = torch.tensor(data[:-1])
        target_ids = torch.tensor(data[1:])
        return input_ids, target_ids

if __name__ == "__main__":
    # Example usage if run directly
    # tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer = BPETokenizer(vocab_size=10000)  # Adjust vocab_size as needed
    dataset = WikiDataset(tokenizer, block_size=128, split='train')
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    print(f"Number of examples in dataset: {len(dataset)}")
    if len(dataset) > 0:
        input_sample, target_sample = dataset[0]
        print(f"Shape of input sample: {input_sample.shape}")
        print(f"Shape of target sample: {target_sample.shape}")

        sample_batch = next(iter(loader))
        print(f"Shape of batch inputs: {sample_batch[0].shape}")
        print(f"Shape of batch targets: {sample_batch[1].shape}")