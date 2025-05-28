import torch
import torch.nn as nn
from models.gpt_mini import GPTMini
from custom_datasets.wiki_dataset import WikiDataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

def train(model, data_loader, optimizer, device):
    """
    Trains the GPTMini model for one epoch.

    Args:
        model: The GPTMini model.
        data_loader: DataLoader for the training data.
        optimizer: Optimizer for model parameters.
        device: Device to train on (e.g., 'cuda', 'cpu').
    """
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0
    num_batches = 0

    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)

        # Calculate loss
        # Reshape logits and y to match CrossEntropyLoss expectation
        loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    print(f"Training Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    # Example usage (requires models/gpt_mini.py and data/dataset.py)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    vocab_size = tokenizer.vocab_size

    model = GPTMini(vocab_size).to(device)
    dataset = WikiDataset(tokenizer, block_size=128, split='train')
    loader = DataLoader(dataset, batch_size=4, shuffle=True) # Use smaller batch size for example
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    print("Running example training epoch...")
    train(model, loader, optimizer, device)