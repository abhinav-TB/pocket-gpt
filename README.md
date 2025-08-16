# pocket-gpt

Pocket GPT is a learning and experimentation playground for building a miniature GPT-style language model from scratch in both C++ (for systems / performance exploration) and Python (for rapid iteration with PyTorch). It also includes a custom Byte Pair Encoding (BPE) tokenizer implementation and a simple training + generation pipeline on the WikiText-2 dataset.

## Contents

- `LLM-Pytorch/` – PyTorch implementation of a small decoder-only transformer (GPTMini) and tooling
  - `models/gpt_mini.py` – Model definition (multi-head causal self-attention + feed-forward blocks)
  - `custom_tokenizers/bpe_tokenizer.py` – Simple, educational BPE tokenizer (train / save / load / encode / decode)
  - `custom_datasets/wiki_dataset.py` – Dataset wrapper over WikiText-2 (splits text into fixed-length token blocks)
  - `utils/trainer.py` – One-epoch training loop helper
  - `utils/generator.py` – Greedy autoregressive generation utility
  - `utils/utils.py` – Reproducibility helper (`set_seed`)
  - `main.py` – Entry point for training or generation
- `cpp/` – (Work-in-progress) C++ experiments (naive + optimized CPU implementations, softmax, attention, OpenMP)
- `env/` – Local Python virtual environment (not tracked ideally; consider adding to `.gitignore`)
- `todo*` – Notes / planning artifacts

> Tip: Keep large virtual environment folders and downloaded datasets out of version control. Add them to `.gitignore` to reduce clutter.

## GPTMini Model Overview

| Component | Shape Notes | Purpose |
|-----------|-------------|---------|
| Token Embedding | `(vocab_size, d_model)` | Learn token representations |
| Positional Embedding | `(max_len, d_model)` | Add sequence position info |
| Multi-Head Causal Self-Attention | Heads = `n_heads`, head dim = `d_model / n_heads` | Context mixing with autoregressive (causal) mask |
| Feed-Forward (per block) | `d_model -> d_ff -> d_model` | Non-linear transformation / feature expansion |
| LayerNorm + Residuals | — | Stabilize training, preserve gradients |
| Output Projection (head) | `(d_model, vocab_size)` | Produces logits per token |

Default config (in `GPTMini` constructor):
- `d_model = 128`
- `n_heads = 4`
- `d_ff = 512`
- `num_layers = 4`
- `max_len = 128`

Output logits shape for batch `B` and sequence length `T`: `(B, T, vocab_size)`.

## Custom BPE Tokenizer (Educational)

Implemented in `custom_tokenizers/bpe_tokenizer.py`:
- Trains by iteratively merging most frequent adjacent symbol pairs until target vocab size
- Supports special tokens: `<unk>`, `<pad>`, `<bos>`, `<eos>`
- Saves / loads JSON (merges + vocab)
- `encode(text, add_special_tokens=True|False)` returns a list of integer IDs
- `decode(ids)` performs a naive reconstruction (current limitation: whitespace fidelity is imperfect)

Limitations / Future improvements:
- Spaces are lost during decode (consider treating space as an explicit token)
- No caching for merge operations (could speed up long inputs)
- No handling of rare / Unicode normalization
- Greedy merge application; not optimized for large vocabularies

If you prefer, you can swap in a Hugging Face tokenizer (e.g. `AutoTokenizer.from_pretrained("gpt2")`) with minor adjustments.

## Dataset Pipeline

`WikiDataset`:
- Loads: `wikitext-2-raw-v1` via `datasets.load_dataset`
- Concatenates all text and tokenizes
- Produces sliding (non-overlapping) windows of length `block_size + 1`
- Returns `(input_ids, target_ids)` where `target_ids` is the input shifted by one

Important: For language modeling, loss is computed over predicting the next token at every position.

## Installation & Environment

It's recommended to create a fresh virtual environment (example uses Windows PowerShell):

```powershell
# (Optional) create environment (Python 3.11+ recommended)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install core dependencies
pip install -r LLM-Pytorch/requirements.txt
```

Add (if missing) packages you need:
```powershell
pip install datasets transformers
```

If you plan to log experiments later:
```powershell
pip install mlflow
```

### CUDA / GPU
- Ensure a CUDA-enabled PyTorch wheel matching your driver: https://pytorch.org/get-started/locally/
- Example (adjust for your CUDA version):
  ```powershell
  pip install torch --index-url https://download.pytorch.org/whl/cu126
  ```
- Verify GPU: 
  ```powershell
  python -c "import torch; print(torch.cuda.is_available())"
  ```

## Quick Start

### 1. (Optional) Train / Load a BPE Tokenizer
Currently `main.py` attempts to load a tokenizer file (`--tokenizer_path`). If it exists, it loads; otherwise it uses an untrained tokenizer (you can extend logic to auto-train and save).

To explicitly train & save (example sketch you can adapt):
```powershell
python - <<'PY'
from custom_tokenizers.bpe_tokenizer import BPETokenizer
from datasets import load_dataset
corpus = "\n\n".join(load_dataset("wikitext","wikitext-2-raw-v1",split='train')["text"])
tok = BPETokenizer(vocab_size=2000)
tok.train(corpus)
tok.save_tokenizer("bpe_tokenizer.json")
PY
```

### 2. Train the Model
```powershell
cd LLM-Pytorch
python main.py --mode train --epochs 2 --batch_size 16 --learning_rate 1e-4 --model_path gptmini.pt --tokenizer_path ..\bpe_tokenizer.json
```

### 3. Generate Text
```powershell
python main.py --mode generate --model_path gptmini.pt --tokenizer_path ..\bpe_tokenizer.json --prompt "The meaning of life is" --max_gen_len 40
```

## Training Details
- Loss: `CrossEntropyLoss` over flattened `(B*T, vocab_size)` logits
- Optimizer: Adam (`lr` configurable)
- No learning rate scheduling yet
- No gradient clipping; consider adding for stability
- No mixed precision yet (AMP could speed up on GPU)

Potential Upgrades:
- Add validation split + early stopping
- Top-k / nucleus sampling for more diverse generation (currently greedy)
- Temperature scaling in generation
- Gradient accumulation for larger effective batch size

## Generation
`utils/generator.py` implements greedy decoding:
- Feeds only the last `max_len` tokens to respect context window
- Repeats for `max_gen_len` steps
- Replace `argmax` with sampling + temperature / top-k for diversity.

## Parameter Counting
Use `count_parameters(model)` (in `gpt_mini.py`) — helpful to budget experiments.

## Known Limitations
| Area | Limitation | Suggested Fix |
|------|------------|---------------|
| Tokenizer decode | Loses original spacing | Add explicit space token or regex pre-tokenizer |
| Training efficiency | No batching overlap or AMP | Use `torch.cuda.amp.autocast` + GradScaler |
| Generation diversity | Greedy only | Implement top-k / nucleus sampling |
| Experiment tracking | MLflow not wired yet | See MLflow plan below |
| Dataset windows | Non-overlapping blocks | Allow stride < block_size for more samples |

## (Planned) MLflow Integration
You asked about adding MLflow. A minimal integration plan:
1. Add `mlflow` to `requirements.txt`
2. Wrap training loop:
   ```python
   import mlflow
   with mlflow.start_run():
       mlflow.log_params({
           'd_model': d_model, 'n_heads': n_heads, 'num_layers': num_layers,
           'vocab_size': vocab_size, 'lr': args.learning_rate, 'epochs': args.epochs
       })
       for epoch in ...:
           loss = train(...)
           mlflow.log_metric('train_loss', loss, step=epoch)
       mlflow.pytorch.log_model(model, 'model')
       mlflow.log_artifact(args.tokenizer_path)
   ```
3. Optionally set tracking URI via `MLFLOW_TRACKING_URI` env var
4. Later: compare runs, export artifacts

## Roadmap
- [ ] Improve BPE decoding (space handling)
- [ ] Add MLflow experiment tracking
- [ ] Add sampling strategies (temperature, top-k, top-p)
- [ ] Add validation + perplexity metric
- [ ] Add mixed precision training
- [ ] Optional Hugging Face tokenizer fallback flag
- [ ] More configurable model hyperparameters via CLI
- [ ] C++ inference prototype bridging to Python

## Troubleshooting
| Issue | Cause | Fix |
|-------|-------|-----|
| CUDA not detected | Mismatch driver / toolkit / torch wheel | Install matching torch wheel; update GPU driver |
| ImportError: datasets | Missing dependency | `pip install datasets` |
| Memory spike | Large vocab or long block size | Reduce vocab_size or `max_len` |
| Decoding unreadable text | BPE missing spaces | Improve decode logic (treat spaces as tokens) |

## Contributing / Extending
This is an educational codebase—feel free to fork and experiment. Ideas:
- Replace attention with FlashAttention (PyTorch 2.x / xFormers)
- Add rotary or ALiBi positional encodings
- Quantize weights for faster inference
- Implement causal mask caching during generation

## License
Add a suitable license (e.g., MIT) if you plan to share or collaborate publicly.

## Acknowledgments
Inspired by the original GPT architecture and countless educational resources (Karpathy's nanoGPT style minimalism, Hugging Face tooling, etc.).

---
Questions or ideas? Open an issue or extend the roadmap section.
