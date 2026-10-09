

words = "low low low lower newest newest widest"

class BPE:
    def __init__(self, num_merges = 5):
        self.num_merges = num_merges
        self.vocab = {}
        self.merges = []

    def counter(self, words):
        counts = {}

        if isinstance(words, str):
            words = words.split()

        for word in words:
            if word in counts:
                counts[word] += 1
            else:
                counts[word] = 1

        return counts

    def split_vocab(self, counts):
        vocab = {}
        for word, freq in counts.items():
            symbols = list(word) + ['</w>']
            vocab[' '.join(symbols)] = freq
        return vocab

    def get_stats(self, vocab):
        pairs = {}
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                pair = (symbols[i], symbols[i+1])
                if pair in pairs:
                    pairs[pair] += freq
                else:
                    pairs[pair] = freq
        return pairs
    
    def merge_vocab(self, pair, vocab):
        new_vocab = {}
        bigram = ' '.join(pair)
        for word, freq in vocab.items():
            new_word = word.replace(bigram, pair[0]+pair[1])
            new_vocab[new_word] =freq
        self.vocab = new_vocab

        return self.vocab

    def train(self, words):
        counts = self.counter(words)
        vocab = self.split_vocab(counts)
        
        for i in range(self.num_merges):
            stats = self.get_stats(vocab)
            pair = max(stats, key=stats.get)
            self.merges.append(pair)
            vocab = self.merge_vocab(pair, vocab)

        self.vocab = vocab
        # 
        self.build_vocab()

        return self.vocab, self.merges

    def build_vocab(self):
        tokens = set()

        for word in self.vocab:
            tokens.update(word.split())

        self.token_to_id = {
            token: i
            for i, token in enumerate(sorted(tokens))
        }

        self.id_to_token = {
            i: token
            for token, i in self.token_to_id.items()
        }

        print('self.token_to_id', self.token_to_id)
        print('self.id_to_token', self.id_to_token)

# python3 bpe.py

bpe = BPE(num_merges = 5)
# print(bpe.train(words))

bpe.train(words)

print('vocab---', bpe.vocab)
print('merges---', bpe.merges)