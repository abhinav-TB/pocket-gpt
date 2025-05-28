import re
from collections import Counter, defaultdict
import json
from dataset import load_dataset

class BPETokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.merges = {}  # Stores the learned merges
        self.vocab = {}   # Stores token to ID mapping
        self.decoder = {} # Stores ID to token mapping
        # Define special tokens
        self.special_tokens = ["<unk>", "<pad>", "<bos>", "<eos>"]
        self.unk_token = "<unk>" # Unknown token
        self.pad_token = "<pad>" # Padding token
        self.bos_token = "<bos>" # Beginning of sentence token
        self.eos_token = "<eos>" # End of sentence token

    def _initialize_vocab_and_decoder(self):
        """Initializes vocab and decoder from merges and special tokens."""
        self.vocab = {token: i for i, token in enumerate(self.special_tokens)}
        # Add individual characters from merges to vocab
        # Merges are (char1, char2) -> merged_char
        # Vocab should contain char1, char2, and merged_char
        all_chars_in_merges = set()
        for pair in self.merges.keys():
            all_chars_in_merges.add(pair[0])
            all_chars_in_merges.add(pair[1])
        
        current_idx = len(self.vocab)
        for char_tuple in sorted(list(self.merges.keys())): # Ensure consistent ordering
            merged_token = "".join(char_tuple)
            if char_tuple[0] not in self.vocab:
                self.vocab[char_tuple[0]] = current_idx
                current_idx +=1
            if char_tuple[1] not in self.vocab:
                self.vocab[char_tuple[1]] = current_idx
                current_idx +=1
            if merged_token not in self.vocab:
                 self.vocab[merged_token] = current_idx
                 current_idx +=1
        
        # Add remaining initial characters (if any, from training)
        # This part is tricky if we only store merges.
        # For simplicity, let's assume the initial vocab is built during training.
        # If loading, we rely on the saved vocab.

        self.decoder = {idx: token for token, idx in self.vocab.items()}

    def get_stats(self, word_freqs):
        """Counts the frequency of character pairs in the corpus (represented by word frequencies)."""
        pairs = defaultdict(int)
        for word, freq in word_freqs.items():
            symbols = list(word) # Split word into characters
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i+1])] += freq
        return pairs

    def merge_pair(self, pair, word_freqs_in):
        """Merges a pair in the corpus (represented by word frequencies)."""
        # pair is a tuple (char1, char2)
        # merged_token is char1char2
        merged_token = "".join(pair)
        word_freqs_out = {}
        for word, freq in word_freqs_in.items():
            # Replace all occurrences of (char1, char2) with merged_token in the word's symbol list
            symbols = list(word)
            i = 0
            new_symbols = []
            while i < len(symbols):
                if i < len(symbols) -1 and (symbols[i], symbols[i+1]) == pair:
                    new_symbols.append(merged_token)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            new_word = "".join(new_symbols) # This is not quite right, BPE operates on symbols
                                          # The 'word' itself becomes a list of symbols/tokens
            word_freqs_out[new_word] = freq # This needs to be rethought for BPE logic
                                            # BPE operates on a list of symbols for each word
        # This function needs to update the representation of words, not just merge strings
        # For now, let's simplify and assume it updates the 'corpus' representation
        # A better way is to update the list of symbols for each word.
        return word_freqs_out # This is a placeholder for a more complex update

    def train(self, corpus_text, vocab_size_target=None):
        """Learns the vocabulary from the corpus text."""
        if vocab_size_target is None:
            vocab_size_target = self.vocab_size

        # 1. Pre-tokenize into words (simple whitespace split for now)
        #    And get initial character vocabulary
        words = corpus_text.split()
        word_freqs = Counter(words)
        
        # Initialize alphabet (all unique characters)
        alphabet = set()
        for word in word_freqs.keys():
            alphabet.update(list(word))
        
        self.vocab = {token: i for i, token in enumerate(self.special_tokens)}
        current_idx = len(self.vocab)
        for char in sorted(list(alphabet)): # Ensure consistent ordering
            if char not in self.vocab:
                self.vocab[char] = current_idx
                current_idx += 1
        
        # Represent words as lists of characters
        # e.g., "hello" -> ["h", "e", "l", "l", "o"]
        # We need to operate on this representation
        split_words = {word: list(word) for word in word_freqs.keys()}

        self.merges = {}
        num_merges = vocab_size_target - len(self.vocab)

        for i in range(num_merges):
            # Recalculate pair stats based on current split_words representation
            pair_stats = defaultdict(int)
            for word, freq in word_freqs.items():
                symbols = split_words[word]
                for j in range(len(symbols) - 1):
                    pair_stats[(symbols[j], symbols[j+1])] += freq
            
            if not pair_stats:
                break
            
            best_pair = max(pair_stats, key=pair_stats.get)
            self.merges[best_pair] = i # Store merge order

            # Update split_words by merging the best_pair
            new_token = "".join(best_pair)
            if new_token not in self.vocab: # Add new merged token to vocab
                self.vocab[new_token] = current_idx
                current_idx += 1

            for word in word_freqs.keys():
                symbols = split_words[word]
                j = 0
                new_symbols = []
                while j < len(symbols):
                    if j < len(symbols) - 1 and (symbols[j], symbols[j+1]) == best_pair:
                        new_symbols.append(new_token)
                        j += 2
                    else:
                        new_symbols.append(symbols[j])
                        j += 1
                split_words[word] = new_symbols
            

            if (i + 1) % 100 == 0: # Progress update
                print(f"Merge {i+1}/{num_merges}")

        self._initialize_vocab_and_decoder() # Finalize vocab and decoder

    def _tokenize_word(self, word_text):
        """Tokenizes a single word using learned merges."""
        if not word_text:
            return []
        
        tokens = list(word_text) # Start with characters

        # Apply merges iteratively
        # A more efficient way is to use the stored merges in order
        # For now, a simpler (but less efficient for many merges) approach:
        while True:
            min_rank_pair = None
            min_rank = float('inf')
            
            # Find the pair with the lowest merge rank (earliest merge)
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i+1])
                if pair in self.merges:
                    rank = self.merges[pair]
                    if rank < min_rank:
                        min_rank = rank
                        min_rank_pair = pair
                        # Store position to merge
                        # This needs to be more robust if multiple occurrences of the same rank pair
                        # For simplicity, take the first one found
                        pair_idx_to_merge = i 
            
            if min_rank_pair is None: # No more mergeable pairs
                break

            # Perform the merge
            merged_token = "".join(min_rank_pair)
            tokens = tokens[:pair_idx_to_merge] + [merged_token] + tokens[pair_idx_to_merge+2:]
            
        return tokens


    def encode(self, text, add_special_tokens=False):
        """Encodes text into a list of token IDs."""
        # Simple whitespace tokenization for words
        words = text.split()
        encoded_ids = []

        if add_special_tokens and self.bos_token in self.vocab:
            encoded_ids.append(self.vocab[self.bos_token])

        for word in words:
            word_tokens = self._tokenize_word(word)
            for token in word_tokens:
                encoded_ids.append(self.vocab.get(token, self.vocab.get(self.unk_token, 0)))
        
        if add_special_tokens and self.eos_token in self.vocab:
            encoded_ids.append(self.vocab[self.eos_token])
            
        return encoded_ids

    def decode(self, token_ids, skip_special_tokens=False):
        """Decodes a list of token IDs back into text."""
        tokens = []
        for token_id in token_ids:
            token = self.decoder.get(token_id, self.unk_token)
            if skip_special_tokens and token in self.special_tokens:
                continue
            tokens.append(token)
        # Joining logic might need adjustment based on how words were split
        # If we tokenized word by word, we might want to join with spaces
        # If it's character/subword level, direct join is fine.
        # Current _tokenize_word produces subwords of original words.
        # We need to reconstruct spaces. This is a common challenge.
        # For now, simple join.
        text = "".join(tokens) # This will lose spaces between original words.
                               # A better approach is to tokenize based on a regex that preserves spaces
                               # or to treat spaces as separate tokens.
        # A common BPE approach is to operate on pre-tokenized words and then join them.
        # Let's assume for now the output is a sequence of subwords that form words.
        # To reconstruct sentences, we'd need to handle spaces better.
        # For example, if spaces were part of the training corpus and merges.
        return text.replace("</w>", " ").strip() # If using a word-end marker like </w>

    def save_tokenizer(self, filepath):
        """Saves the tokenizer's merges and vocab to a file."""
        # Convert tuple keys in merges to strings for JSON serialization
        serializable_merges = {f"{k[0]}_{k[1]}": v for k, v in self.merges.items()}
        data = {
            "vocab_size": self.vocab_size,
            "merges": serializable_merges,
            "vocab": self.vocab, # vocab already has string keys
            "special_tokens": self.special_tokens
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Tokenizer saved to {filepath}")

    @classmethod
    def from_file(cls, filepath):
        """Loads the tokenizer from a file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tokenizer = cls(vocab_size=data["vocab_size"])
        # Convert string keys back to tuples for merges
        tokenizer.merges = {tuple(k.split('_', 1)): v for k, v in data["merges"].items()}
        tokenizer.vocab = data["vocab"]
        tokenizer.special_tokens = data.get("special_tokens", ["<unk>", "<pad>", "<bos>", "<eos>"]) # Handle older files
        tokenizer.unk_token = tokenizer.special_tokens[0] if len(tokenizer.special_tokens) > 0 else "<unk>"
        tokenizer.pad_token = tokenizer.special_tokens[1] if len(tokenizer.special_tokens) > 1 else "<pad>"
        tokenizer.bos_token = tokenizer.special_tokens[2] if len(tokenizer.special_tokens) > 2 else "<bos>"
        tokenizer.eos_token = tokenizer.special_tokens[3] if len(tokenizer.special_tokens) > 3 else "<eos>"
        
        tokenizer._initialize_vocab_and_decoder() # Rebuild decoder
        print(f"Tokenizer loaded from {filepath}")
        return tokenizer
    

# Example usage:
if __name__ == '__main__':
    # corpus = (
    #     "This is a sample corpus for training a BPE tokenizer. "
    #     "BPE is a subword tokenization algorithm. "
    #     "It iteratively replaces the most frequent pair of bytes "
    #     "in a sequence with a single, unused byte. "
    #     "hello world how are you doing today hello again"
    # )

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split='train')
    corpus = "\n\n".join(dataset["text"])  # Join all text into a single string
    
    # --- Training ---
    print("Training tokenizer...")
    tokenizer = BPETokenizer(vocab_size=2000) # Smaller vocab for quick example
    # tokenizer.train(corpus)
    
    # # --- Save the tokenizer ---
    # tokenizer.save_tokenizer("bpe_custom_tokenizer.json")
    
    # --- Load the tokenizer ---
    print("\nLoading tokenizer from file...")
    loaded_tokenizer = BPETokenizer.from_file("bpe_tokenizer_wiki_2000.json")

    # --- Test Encoding and Decoding ---
    test_sentence = "hello world this is a test of the BPE algorithm"
    print(f"\nOriginal: {test_sentence}")

    encoded_ids = loaded_tokenizer.encode(test_sentence, add_special_tokens=True)
    print(f"Encoded IDs: {encoded_ids}")

    decoded_text = loaded_tokenizer.decode(encoded_ids, skip_special_tokens=False)
    print(f"Decoded Text (from loaded): {decoded_text}")



    print("\nVocabulary sample (first 20):")
    for i, (token, token_id) in enumerate(loaded_tokenizer.vocab.items()):
        if i >= 20: break
        print(f"'{token}': {token_id}")

    print("\nMerges sample (first 10):")
    for i, (pair, rank) in enumerate(loaded_tokenizer.merges.items()):
        if i >= 10: break
        print(f"{pair}: {rank}")