import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

@torch.no_grad()
def generate(model, start_seq, max_len, device):
    """
    Generates text using the trained GPTMini model.

    Args:
        model: The GPTMini model.
        start_seq: Initial sequence of token IDs (torch.Tensor).
        max_len: Maximum number of tokens to generate.
        device: Device to perform inference on (e.g., 'cuda', 'cpu').

    Returns:
        torch.Tensor: The generated sequence of token IDs.
    """
    model.eval()
    # Ensure start_seq is on the correct device and is a 1D tensor
    start_seq = start_seq.to(device)
    if start_seq.dim() == 0:
        start_seq = start_seq.unsqueeze(0)
    elif start_seq.dim() > 1:
        start_seq = start_seq.squeeze() # Handle potential batch dim from tokenizer

    out = start_seq.clone()

    for _ in range(max_len):
        # The model expects input of shape (batch_size, sequence_length)
        # For generation, we process one sequence (batch_size=1)
        # Take the last 'model.max_len' tokens if the sequence exceeds max_len
        input_seq = out[-model.max_len:].unsqueeze(0)

        logits = model(input_seq)

        # Get the logits for the last token in the sequence
        logits_last_token = logits[:, -1, :]

        # Get the next token using argmax (greedy decoding)
        next_token = torch.argmax(logits_last_token, dim=-1)

        # Append the generated token to the output sequence
        out = torch.cat([out, next_token], dim=0)

    return out

if __name__ == "__main__":
    # Example usage (requires models/gpt_mini.py)
    from models.gpt_mini import GPTMini

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    vocab_size = tokenizer.vocab_size

    # Create a dummy model (you would load a trained one here)
    model = GPTMini(vocab_size).to(device)
    # Example: Load a trained model
    # model.load_state_dict(torch.load("path/to/your/trained_model.pth", map_location=device))

    prompt = "The meaning of life is"
    input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'][0].to(device)

    print(f"Generating text from prompt: '{prompt}'")
    generated_ids = generate(model, input_ids, max_len=20, device=device)
    generated_text = tokenizer.decode(generated_ids.cpu().tolist())
    print("Generated:", generated_text)