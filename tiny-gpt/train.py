"""Tiny character-level GPT. Usage:
    python train.py --data data/my_stories.txt --steps 3000
"""
import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/my_stories.txt")
parser.add_argument("--steps", type=int, default=3000)
parser.add_argument("--prompt", default="")
parser.add_argument("--length", type=int, default=600)
args = parser.parse_args()

torch.manual_seed(1337)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", device)

# ---- settings (smaller without a GPU) ----
if device == "cuda":
    batch_size, block_size = 64, 128
    n_embd, n_head, n_layer = 192, 4, 4
else:
    batch_size, block_size = 32, 64
    n_embd, n_head, n_layer = 64, 4, 2
dropout = 0.2
learning_rate = 1e-3

# ---- data ----
text = open(args.data, "r", encoding="utf-8").read()
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda l: "".join(itos[i] for i in l)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

# ---- model ----
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = self.dropout(F.softmax(wei, dim=-1))
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = MultiHeadAttention(n_head, n_embd // n_head)
        self.ff = FeedForward()
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        return x + self.ff(self.ln2(x))

class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_emb(idx) + self.pos_emb(torch.arange(T, device=device))
        logits = self.lm_head(self.ln_f(self.blocks(x)))
        if targets is None:
            return logits, None
        return logits, F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -block_size:])
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx

@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = [model(*get_batch(split))[1].item() for _ in range(30)]
        out[split] = round(sum(losses) / len(losses), 3)
    model.train()
    return out

# ---- train ----
model = TinyGPT().to(device)
print(f"Model has {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for step in range(args.steps + 1):
    if step % 500 == 0:
        print(f"step {step}: {estimate_loss(model)}")
    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), "model.pt")

# ---- generate ----
start = torch.tensor([encode(args.prompt) or [0]], dtype=torch.long, device=device)
print("\n--- Generated text ---")
print(decode(model.generate(start, args.length)[0].tolist()))
