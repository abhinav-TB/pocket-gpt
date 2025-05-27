import re
from collections import Counter

class BPETokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.encoder = {}
        self.decoder = {}

    def train(self, corpus):
        """Learns the vocabulary from the corpus."""
        # 1. Initialize vocabulary with individual characters
        self.vocab = {char: idx for idx, char in enumerate(set(corpus))}
        next_idx = len(self.vocab)
        corpus = "".join(corpus)

        # 2. Iteratively merge the most frequent pairs until vocab_size is reached
        while len(self.vocab) < self.vocab_size:
            stats = self.get_stats(corpus)
            if not stats:
                break
            best_pair = max(stats, key=stats.get)
            corpus = self.merge_vocab(best_pair, corpus)
            self.vocab[best_pair[0] + best_pair[1]] = next_idx
            next_idx += 1

        # 3. Create encoder and decoder dictionaries
        self.encoder = self.vocab
        self.decoder = {idx: token for token, idx in self.vocab.items()}

    def get_stats(self, corpus):
        """Counts the frequency of character pairs in the corpus."""
        pairs = []
        for i in range(len(corpus) - 1):
            pairs.append(corpus[i:i+2])
        return Counter(pairs)

    def merge_vocab(self, pair, corpus):
        """Merges a pair in the corpus."""
        new_corpus = corpus.replace(pair, pair[0] + pair[1])
        return new_corpus

    def encode(self, text):
        """Encodes text into a list of token IDs."""
        # 1. Tokenize the text into words
        tokens = list(text)  # Split into characters

        # 2. Apply vocabulary to each word
        ids = []
        for token in tokens:
            if token in self.encoder:
                ids.append(self.encoder[token])
            else:
                ids.append(self.encoder.get("<unk>", 0))  # Handle unknown tokens

        # 3. Return list of token IDs
        return ids

    def decode(self, tokens):
        """Decodes a list of token IDs back into text."""
        # 1. Convert token IDs to strings
        text = "".join([self.decoder.get(token, "<unk>") for token in tokens])
        return text

# Example usage:
if __name__ == '__main__':
    corpus = "This is a sample corpus for training a BPE tokenizer. BPE is a subword tokenization algorithm."
    tokenizer = BPETokenizer(vocab_size=50)
    tokenizer.train(corpus)
    encoded = tokenizer.encode("This is a test.")
    decoded = tokenizer.decode(encoded)
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")