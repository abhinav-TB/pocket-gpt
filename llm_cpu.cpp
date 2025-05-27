#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <iostream>
#include <cassert>

using Vector = std::vector<float>;
using Tensor3D = std::vector<std::vector<std::vector<float>>>;
using Tensor4D = std::vector<std::vector<std::vector<std::vector<float>>>>;

// Utility: in-place softmax
void softmax(Vector& v) {
    float maxv = *std::max_element(v.begin(), v.end());
    float sum = 0.0f;
    for (auto& x : v) { x = std::exp(x - maxv); sum += x; }
    for (auto& x : v) x /= sum;
}

// Simple Linear layer
struct Linear {
    int in_features, out_features;
    std::vector<Vector> weight; // [out_features][in_features]
    Vector bias;                // [out_features]

    Linear(int in_f, int out_f)
        : in_features(in_f), out_features(out_f),
          weight(out_f, Vector(in_f)), bias(out_f, 0.0f) {
        std::mt19937 gen(42);
        std::normal_distribution<float> dist(0, std::sqrt(2.0f / in_f));
        for (auto& row : weight)
            for (auto& w : row)
                w = dist(gen);
    }

    Vector forward(const Vector& x) const {
        assert((int)x.size() == in_features);
        Vector y(out_features);
        for (int i = 0; i < out_features; ++i) {
            float sum = bias[i];
            for (int j = 0; j < in_features; ++j)
                sum += weight[i][j] * x[j];
            y[i] = sum;
        }
        return y;
    }
};

// Layer normalization
struct LayerNorm {
    int dim;
    float eps;
    Vector gamma, beta;

    LayerNorm(int d, float e = 1e-5f)
        : dim(d), eps(e), gamma(d, 1.0f), beta(d, 0.0f) {}

    Vector forward(const Vector& x) const {
        assert((int)x.size() == dim);
        float mean = 0.0f, var = 0.0f;
        for (auto v : x) mean += v;
        mean /= dim;
        for (auto v : x) var += (v - mean) * (v - mean);
        var /= dim;
        Vector out(dim);
        for (int i = 0; i < dim; ++i)
            out[i] = ((x[i] - mean) / std::sqrt(var + eps)) * gamma[i] + beta[i];
        return out;
    }
};

// Causal Self-Attention
struct CausalSelfAttention {
    int d_model, n_heads, d_k;
    Linear qkv_proj, out_proj;

    CausalSelfAttention(int dm, int nh)
        : d_model(dm), n_heads(nh), d_k(dm / nh),
          qkv_proj(dm, 3 * dm), out_proj(dm, dm) {}

    Tensor3D forward(const Tensor3D& x) const {
        int B = x.size();
        int T = x[0].size();
        // Project to QKV
        Tensor3D qkv(B, std::vector<std::vector<float>>(T, Vector(3 * d_model)));
        for (int b = 0; b < B; ++b)
            for (int t = 0; t < T; ++t)
                qkv[b][t] = qkv_proj.forward(x[b][t]);

        // Split into heads: Q, K, V
        Tensor4D Q(B, std::vector<std::vector<std::vector<float>>>(n_heads, std::vector<std::vector<float>>(T, Vector(d_k))));
        Tensor4D K = Q, V = Q;
        for (int b = 0; b < B; ++b)
            for (int t = 0; t < T; ++t) {
                const Vector& vec = qkv[b][t];
                for (int h = 0; h < n_heads; ++h) {
                    for (int i = 0; i < d_k; ++i) {
                        Q[b][h][t][i] = vec[h * d_k + i];
                        K[b][h][t][i] = vec[d_model + h * d_k + i];
                        V[b][h][t][i] = vec[2 * d_model + h * d_k + i];
                    }
                }
            }

        // Scaled dot-product with causal mask
        Tensor4D scores(B, std::vector<std::vector<std::vector<float>>>(n_heads, std::vector<std::vector<float>>(T, Vector(T))));
        float inv_sqrt = 1.0f / std::sqrt((float)d_k);
        for (int b = 0; b < B; ++b)
            for (int h = 0; h < n_heads; ++h)
                for (int i = 0; i < T; ++i) {
                    for (int j = 0; j < T; ++j) {
                        float dot = 0.0f;
                        for (int k_i = 0; k_i < d_k; ++k_i)
                            dot += Q[b][h][i][k_i] * K[b][h][j][k_i];
                        // Mask future positions
                        scores[b][h][i][j] = (j <= i) ? dot * inv_sqrt : -1e9f;
                    }
                    softmax(scores[b][h][i]);
                }

        // Weighted sum to get context
        Tensor4D context = V; // reuse structure
        for (int b = 0; b < B; ++b)
            for (int h = 0; h < n_heads; ++h)
                for (int i = 0; i < T; ++i) {
                    for (int k_i = 0; k_i < d_k; ++k_i) {
                        float sum = 0.0f;
                        for (int j = 0; j < T; ++j)
                            sum += scores[b][h][i][j] * V[b][h][j][k_i];
                        context[b][h][i][k_i] = sum;
                    }
                }

        // Concatenate heads and final linear
        Tensor3D out(B, std::vector<std::vector<float>>(T, Vector(d_model)));
        for (int b = 0; b < B; ++b)
            for (int t = 0; t < T; ++t) {
                Vector cat(d_model);
                for (int h = 0; h < n_heads; ++h)
                    for (int i = 0; i < d_k; ++i)
                        cat[h * d_k + i] = context[b][h][t][i];
                out[b][t] = out_proj.forward(cat);
            }
        return out;
    }
};

// Position-wise Feed-Forward
struct FeedForward {
    Linear fc1, fc2;
    FeedForward(int dm, int dff) : fc1(dm, dff), fc2(dff, dm) {}

    Tensor3D forward(const Tensor3D& x) const {
        int B = x.size(), T = x[0].size();
        Tensor3D out(B, std::vector<std::vector<float>>(T, Vector(x[0][0].size())));
        for (int b = 0; b < B; ++b)
            for (int t = 0; t < T; ++t) {
                Vector v = fc1.forward(x[b][t]);
                for (auto& e : v) e = std::max(0.0f, e);
                out[b][t] = fc2.forward(v);
            }
        return out;
    }
};

// Transformer Decoder Block
struct DecoderBlock {
    CausalSelfAttention attn;
    FeedForward ff;
    LayerNorm norm1, norm2;

    DecoderBlock(int dm, int nh, int dff)
        : attn(dm, nh), ff(dm, dff), norm1(dm), norm2(dm) {}

    Tensor3D forward(const Tensor3D& x) const {
        Tensor3D y1 = attn.forward(x);
        int B = x.size(), T = x[0].size(), C = x[0][0].size();
        Tensor3D x1 = x;
        for (int b = 0; b < B; ++b)
            for (int t = 0; t < T; ++t) {
                for (int i = 0; i < C; ++i)
                    x1[b][t][i] += y1[b][t][i];
                x1[b][t] = norm1.forward(x1[b][t]);
            }
        Tensor3D y2 = ff.forward(x1);
        Tensor3D x2 = x1;
        for (int b = 0; b < B; ++b)
            for (int t = 0; t < T; ++t) {
                for (int i = 0; i < C; ++i)
                    x2[b][t][i] += y2[b][t][i];
                x2[b][t] = norm2.forward(x2[b][t]);
            }
        return x2;
    }
};

// GPTMini Model
struct GPTMini {
    int vocab_size, d_model;
    std::vector<Vector> token_emb, pos_emb;
    std::vector<DecoderBlock> blocks;
    LayerNorm ln_f;
    Linear head;

    GPTMini(int vs, int dm=128, int nh=4, int dff=512, int nl=4, int max_len=128)
        : vocab_size(vs), d_model(dm),
          token_emb(vs, Vector(dm)), pos_emb(max_len, Vector(dm)),
          ln_f(dm), head(dm, vs) {
        std::mt19937 gen(123);
        std::normal_distribution<float> dist(0, 1.0f/std::sqrt(dm));
        for (auto& emb : token_emb)
            for (auto& e : emb) e = dist(gen);
        for (auto& emb : pos_emb)
            for (auto& e : emb) e = dist(gen);
        for (int i = 0; i < nl; ++i)
            blocks.emplace_back(dm, nh, dff);
    }

    Tensor3D forward(const std::vector<std::vector<int>>& idx) const {
        int B = idx.size(), T = idx[0].size();
        Tensor3D x(B, std::vector<std::vector<float>>(T, Vector(d_model)));
        for (int b = 0; b < B; ++b)
            for (int t = 0; t < T; ++t)
                for (int i = 0; i < d_model; ++i)
                    x[b][t][i] = token_emb[idx[b][t]][i] + pos_emb[t][i];
        for (const auto& block : blocks)
            x = block.forward(x);
        for (int b = 0; b < B; ++b)
            for (int t = 0; t < T; ++t)
                x[b][t] = ln_f.forward(x[b][t]);

        // logits: [B][T][vocab_size]
        Tensor3D logits(B, std::vector<std::vector<float>>(T, Vector(vocab_size)));
        for (int b = 0; b < B; ++b)
            for (int t = 0; t < T; ++t)
                logits[b][t] = head.forward(x[b][t]);
        return logits;
    }
};

int main() {
    int vocab_size = 1000;
    GPTMini model(vocab_size);
    int B = 2, T = 10;
    std::vector<std::vector<int>> idx(B, std::vector<int>(T));
    std::mt19937 gen(7);
    std::uniform_int_distribution<int> dist(0, vocab_size - 1);
    for (int b = 0; b < B; ++b)
        for (int t = 0; t < T; ++t)
            idx[b][t] = dist(gen);
    Tensor3D logits = model.forward(idx);
    std::cout << "Logits shape: (" << logits.size() << ", "
              << logits[0].size() << ", "
              << logits[0][0].size() << ")\n";
    return 0;
}
