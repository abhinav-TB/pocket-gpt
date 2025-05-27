import math
import numpy as np

# Utility function: softmax
def softmax(x, axis=-1):
    """Compute softmax values for each set of scores in x along specified axis."""
    # Subtract max for numerical stability
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

class Linear:
    def __init__(self, in_features, out_features, bias=True):
        self.weight = np.random.normal(0, 0.02, (out_features, in_features))
        self.bias = np.zeros(out_features) if bias else None
        
    def __call__(self, x):
        output = np.dot(x, self.weight.T)
        if self.bias is not None:
            output += self.bias
        return output

class LayerNorm:
    def __init__(self, normalized_shape, eps=1e-5):
        self.eps = eps
        self.gamma = np.ones(normalized_shape)
        self.beta = np.zeros(normalized_shape)
        
    def __call__(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta
    
class Embedding:
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = np.random.normal(0, 0.02, (num_embeddings, embedding_dim))
        
    def __call__(self, idx):
        return self.weight[idx]

class CausalSelfAttention:
    def __init__(self, d_model, n_heads):
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        
        self.qkv_proj = Linear(d_model, 3 * d_model)
        self.out_proj = Linear(d_model, d_model)
        
    def __call__(self, x):
        B, T, C = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # (B, T, 3*C)
        qkv = qkv.reshape(B, T, self.n_heads, 3 * self.d_k)
        qkv = np.transpose(qkv, (0, 2, 1, 3))  # (B, n_heads, T, 3*d_k)
        
        # Split into q, k, v
        chunk_size = qkv.shape[-1] // 3
        q = qkv[..., :chunk_size]
        k = qkv[..., chunk_size:2*chunk_size]
        v = qkv[..., 2*chunk_size:]
        
        # Compute attention scores
        scores = np.matmul(q, np.transpose(k, (0, 1, 3, 2)))  # (B, n_heads, T, T)
        scores = scores / math.sqrt(self.d_k)
        
        # Create causal mask and apply
        mask = np.tril(np.ones((T, T)))
        mask = np.broadcast_to(mask[np.newaxis, np.newaxis, ...], scores.shape)
        scores = np.where(mask == 0, -1e9, scores)
        
        # Apply softmax and compute context
        attn = softmax(scores, axis=-1)  # (B, n_heads, T, T)
        context = np.matmul(attn, v)  # (B, n_heads, T, d_k)
        
        # Reshape back
        context = np.transpose(context, (0, 2, 1, 3)).reshape(B, T, C)
        return self.out_proj(context)

class FeedForward:
    def __init__(self, d_model, d_ff):
        self.fc1 = Linear(d_model, d_ff)
        self.fc2 = Linear(d_ff, d_model)
        
    def __call__(self, x):
        return self.fc2(np.maximum(self.fc1(x), 0))  # ReLU activation

class DecoderBlock:
    def __init__(self, d_model, n_heads, d_ff):
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        
    def __call__(self, x):
        x = self.norm1(x + self.attn(x))  # Residual connection
        x = self.norm2(x + self.ff(x))    # Residual connection
        return x

class GPTMini:
    def __init__(self, vocab_size, d_model=128, n_heads=4, d_ff=512, num_layers=4, max_len=128):
        self.token_emb = Embedding(vocab_size, d_model)
        self.pos_emb = Embedding(max_len, d_model)
        
        self.blocks = []
        for _ in range(num_layers):
            self.blocks.append(DecoderBlock(d_model, n_heads, d_ff))
            
        self.ln_f = LayerNorm(d_model)
        self.head = Linear(d_model, vocab_size, bias=False)
        
    def __call__(self, idx):
        B, T = idx.shape
        pos = np.arange(T)[np.newaxis, :]  # (1, T)
        
        # Combine token and position embeddings
        x = self.token_emb(idx) + self.pos_emb(pos)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
            
        # Final layer norm and prediction head
        x = self.ln_f(x)
        return self.head(x)  # (B, T, vocab_size)

# Example usage
if __name__ == "__main__":
    vocab_size = 1000
    model = GPTMini(vocab_size)
    idx = np.random.randint(0, vocab_size, (2, 10))
    logits = model(idx)
    print(logits.shape)  # Should be (2, 10, 1000)