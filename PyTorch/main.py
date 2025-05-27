import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import argparse
import os

from gpt_mini import GPTMini
from dataset import RealTextDataset
from trainer import train
from generator import generate
from utils import set_seed

def main():
    parser = argparse.ArgumentParser(description="Train or generate text with GPTMini.")
    parser.add_argument("--mode", type=str, choices=["train", "generate"], required=True,
                        help="Mode to run: 'train' or 'generate'")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to load or save the model state dict.")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Training batch size.")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for training.")
    parser.add_argument("--max_gen_len", type=int, default=30,
                        help="Maximum length of generated text.")
    parser.add_argument("--prompt", type=str, default="The meaning of life is",
                        help="Prompt for text generation.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    vocab_size = tokenizer.vocab_size

    model = GPTMini(vocab_size).to(device)

    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading model from {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=device))

    if args.mode == "train":
        print("Running in training mode.")
        dataset = RealTextDataset(tokenizer, block_size=model.max_len, split='train') # Use model's max_len
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        print("Starting training...")
        for epoch in range(args.epochs):
            print(f"Epoch {epoch+1}/{args.epochs}")
            train(model, loader, optimizer, device)

        if args.model_path:
            print(f"Saving trained model to {args.model_path}")
            torch.save(model.state_dict(), args.model_path)

    elif args.mode == "generate":
        print("Running in generation mode.")
        if not args.model_path or not os.path.exists(args.model_path):
            print("Error: Model path is required for generation and must exist.")
            return

        prompt_ids = tokenizer(args.prompt, return_tensors='pt')['input_ids'][0].to(device)

        print(f"Generating text from prompt: '{args.prompt}'")
        generated_ids = generate(model, prompt_ids, max_len=args.max_gen_len, device=device)
        generated_text = tokenizer.decode(generated_ids.cpu().tolist())
        print("Generated:", generated_text)

if __name__ == "__main__":
    main()