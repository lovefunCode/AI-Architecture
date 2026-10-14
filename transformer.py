
import copy;
import torch
import torch.nn as nn   
import math

def attention(query, key, value, mask = None, dropout = None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1))/math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e-9)
    p_attn = scores.softmax(dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout = 0.1):
        super(MultiHeadedAttention, self).__init__()
        self.d_k = d_model // h 
        self.h = h
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.attn = None
        self.linears = clones(nn.Linear(d_model, d_model), 4)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nBatches = query.size(0)

        # 1. do all the linear projection to the  query, key, value
        query, key, value = [
            lin(x).view(nBatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears , (query, key, value))
        ]

        # 2 apply attention on all the projected vectors in batch
        x, self.attn = attention(query, key, value, mask=mask, dropout = self.dropout)

        # 3. concat and linear
        x = (x.transpose(1,2).contiguous().view(nBatches, -1, self.h * self.d_k))

        del query
        del key
        del value
        return self.linears[-1](x)
    


# FFN(x) = max(0, w1* x + b1) * w2 + b2
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout = None):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))
    

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
    
# normalize across features for 1 sample
class LayerNorm(nn.Module):
    def __init__(self, features, eps = 1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim = True)
        std = x.std(-1, keepdim = True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x : self.self_attn(x, x, x, mask))
        return  self.sublayer[1](x, self.feed_forward)

class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
        self.size = size 

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, src_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, memory, memory, tgt_mask))
        return self.sublayer[2](x, self.feed_forward)


class Embedding(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embedding, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
    
    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim = -1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p = dropout)

        #compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000)/d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cons(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


def make_model(src_vocab, tgt_vocab, d_model = 512, h=8, d_ff = 2048, N = 6, dropout = 0.1):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embedding(d_model, src_vocab), c(position)),
        nn.Sequential(Embedding(d_model, tgt_vocab), c(position)), 
        Generator(d_model, tgt_vocab)            
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model

model = make_model(
    src_vocab=10000,
    tgt_vocab=10000,
)


class TrainState:
    step: int = 0
    accum_step: int = 0
    samples: int = 0
    tokens: int = 0

def run_epoch(data_iter, model , loss_compute, optimizer, scheduler, mode="train", accum_iter = 1, train_state=train_state()):
    start_time = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_aacum = 0
    for i, batch in enumerate(data_iter):
        out = mode.forward(
            batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
        )
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.nTokens)
        if mode == 'train' or mode == 'train+log':
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.nTokens
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_aacum += 1
                train_state.accum_step += 1
            scheduler.step()
        total_loss += loss
        total_tokens += batch.nTokens
        tokens += batch.nTokens
        if i % 40 == 1 and (mode == 'train' or mode == "train+log"):
            lr = optimizer.param_groups[0]['lr']
            elapsed = time.time() - start
            print(
                (
                    'Epoch step: %6d | Accumulation Step : %3d | Loss: %6.2f'
                    + "| Tokens/Sec: %7.1f | Learning Rate: %6.1e"
                )
                %(i, n_aacum, loss/batch.nTokens, tokens/elapsed, lr)
            )

            start = time.time()
            tokens = 0

        del loss
        del loss_mode
    return total_loss / total_tokens, train_state



src = torch.randint(0, 10000, (2, 10))
tgt = torch.randint(0, 10000, (2, 10))

src_mask = torch.ones(2, 1, 10).bool()
tgt_mask = torch.ones(2, 10, 10).bool()

print("Running forward pass...")

out = model(src, tgt, src_mask, tgt_mask)

print("Output shape:", out.shape)


# python3 transformer.py
