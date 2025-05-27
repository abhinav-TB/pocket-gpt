import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads # 128/4 = 32 , this is the size of each q,k,v matrix
        self.n_heads = n_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.size() # (16*128*128)
        qkv = self.qkv_proj(x) #(16*128*384) (16*128*(3*128))
        qkv = qkv.reshape(B, T, self.n_heads, 3 * self.d_k) #(16*128*4*(3*32)) (16*128*4*96)
        qkv = qkv.permute(0, 2, 1, 3) #(16*4*128*96)
        q, k, v = qkv.chunk(3, dim=-1) #(16*4*128*32) (16*4*128*32) (16*4*128*32)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k) #(16*4*128*128) -2 , -1 exchanges last two dimensions
        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0) #(1*1*128*128) a lower traingular matrix
        scores = scores.masked_fill(mask == 0, float('-inf')) #(16*4*128*128)
        attn = F.softmax(scores, dim=-1) #(16*4*128*128)

        context = attn @ v #(16*4*128*32)
        context = context.transpose(1, 2).reshape(B, T, C) #(16*128*128)
        return self.out_proj(context) #(16*128*128)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x))) #16*128*128

class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.norm1(x + self.attn(x)) #(16*128*128) adding for resedual connection
        x = self.norm2(x + self.ff(x)) #(16*16*128)
        return x

class GPTMini(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, d_ff=512, num_layers=4, max_len=128):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([DecoderBlock(d_model, n_heads, d_ff) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.size()  # 16 * 128
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0) #(1*128)
        x = self.token_emb(idx) + self.pos_emb(pos)  # (16*128*128) + (1*128*128)  = (16*128*128)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x) #16*128*50257(vocabsize)

# Example: usage with dummy inputs
if __name__ == "__main__":
    vocab_size = 1000
    model = GPTMini(vocab_size)
    idx = torch.randint(0, vocab_size, (16, 128))
    logits = model(idx)
    print(logits.shape)  # (B, T, vocab_size)
