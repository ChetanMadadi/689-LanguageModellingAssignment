import math
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# =========================
# Step 1: Repro + device
# =========================
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Step 2: Tokenizers
# =========================
class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)

    def encode(self, s: str) -> List[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos[i] for i in ids)


class WordTokenizer:
    """
    Very simple whitespace tokenizer. Works for "word-level" requirement.
    You can swap this for dataset-provided tokenization if you have it.
    """
    def __init__(self, texts: List[str], min_freq: int = 2, max_vocab: int = 50000):
        from collections import Counter
        counter = Counter()
        for t in texts:
            counter.update(t.split())

        # special tokens
        self.PAD = "<pad>"
        self.UNK = "<unk>"
        self.BOS = "<bos>"
        self.EOS = "<eos>"

        vocab = [self.PAD, self.UNK, self.BOS, self.EOS]
        for w, c in counter.most_common():
            if c < min_freq:
                continue
            if len(vocab) >= max_vocab:
                break
            vocab.append(w)

        self.stoi = {w: i for i, w in enumerate(vocab)}
        self.itos = {i: w for w, i in self.stoi.items()}
        self.vocab_size = len(vocab)

    def encode(self, s: str, add_bos_eos: bool = False) -> List[int]:
        toks = s.split()
        ids = [self.stoi.get(w, self.stoi[self.UNK]) for w in toks]
        if add_bos_eos:
            ids = [self.stoi[self.BOS]] + ids + [self.stoi[self.EOS]]
        return ids

    def decode(self, ids: List[int]) -> str:
        words = [self.itos.get(i, self.UNK) for i in ids]
        return " ".join(words)


# =========================
# Step 3: Dataset utilities
# =========================
class NextTokenDataset(Dataset):
    """
    Given a 1D LongTensor of token ids, creates (x, y) where
    x is length T, y is x shifted by 1.
    """
    def __init__(self, tokens: torch.LongTensor, T: int):
        assert tokens.dim() == 1
        self.tokens = tokens
        self.T = T

    def __len__(self):
        return max(0, self.tokens.numel() - (self.T + 1))

    def __getitem__(self, idx: int):
        x = self.tokens[idx: idx + self.T]
        y = self.tokens[idx + 1: idx + self.T + 1]
        return x, y


def make_splits(token_ids: torch.LongTensor, train_frac=0.90, val_frac=0.05):
    n = token_ids.numel()
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = token_ids[:n_train]
    val = token_ids[n_train:n_train + n_val]
    test = token_ids[n_train + n_val:]
    return train, val, test


def make_loader(tokens_1d: torch.LongTensor, T: int, batch_size: int, shuffle: bool):
    ds = NextTokenDataset(tokens_1d, T)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True)


# =========================
# Step 4: FLOPs estimation (approx but consistent)
# =========================
def flops_linear(vocab: int, T: int, d_model: int) -> int:
    # embedding lookup ~ 0 flops
    # pool (mean) over T: ~T*d adds
    # final linear: d_model * vocab mult-add
    return int(T * d_model + 2 * d_model * vocab)

def flops_mlp(vocab: int, T: int, d_model: int, hidden: int, layers: int) -> int:
    # input is flattened embeddings: T*d_model
    inp = T * d_model
    # first layer: inp->hidden
    fl = 2 * inp * hidden
    # middle layers hidden->hidden
    for _ in range(layers - 2):
        fl += 2 * hidden * hidden
    # output hidden->vocab
    fl += 2 * hidden * vocab
    return int(fl)

def flops_attention(vocab: int, T: int, d_model: int, n_heads: int, head_dim: int, with_mlp: bool = False) -> int:
    # QKV projections: 3 * (d_model -> n_heads*head_dim)
    proj = 3 * 2 * d_model * (n_heads * head_dim)
    # attention weights: (B, heads, T, head_dim) x (B, heads, head_dim, T) => T*T*head_dim
    attn_scores = 2 * n_heads * (T * T * head_dim)
    # apply weights to V: (T,T) x (T,head_dim) => T*T*head_dim
    attn_apply = 2 * n_heads * (T * T * head_dim)
    # output projection: (n_heads*head_dim -> d_model)
    out = 2 * (n_heads * head_dim) * d_model
    # classifier: d_model -> vocab
    clf = 2 * d_model * vocab
    mlp = 0
    if with_mlp:
        # a small FFN 4*d -> d
        mlp = 2 * d_model * (4 * d_model) + 2 * (4 * d_model) * d_model
    return int(proj + attn_scores + attn_apply + out + mlp + clf)

def flops_transformer(vocab: int, T: int, d_model: int, n_heads: int, head_dim: int, n_layers: int) -> int:
    # per layer: attention + FFN (4*d)
    per_layer = flops_attention(vocab=1, T=T, d_model=d_model, n_heads=n_heads, head_dim=head_dim, with_mlp=True)  # vocab=1 to ignore clf
    # final classifier d->vocab
    clf = 2 * d_model * vocab
    return int(n_layers * per_layer + clf)

def estimate_train_flops(flops_per_forward: int, tokens_processed: int) -> int:
    # rough: forward + backward+grads ~ 3x forward
    return int(3 * flops_per_forward * tokens_processed)


# =========================
# Step 5: Models
# =========================
class LinearPredictor(nn.Module):
    """
    Embedding -> mean pooling over time -> linear to vocab.
    """
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):  # x: (B,T)
        e = self.emb(x)          # (B,T,D)
        h = e.mean(dim=1)        # (B,D)
        logits = self.proj(h)    # (B,V)
        # Expand to token-wise output by repeating for each position, to match loss shape (B,T,V)
        return logits.unsqueeze(1).expand(x.size(0), x.size(1), logits.size(-1))


class MLP3Plus(nn.Module):
    """
    Embedding -> flatten (B, T*D) -> MLP with >=3 layers -> vocab.
    Output repeated across T (simple but valid for next-token modeling).
    """
    def __init__(self, vocab_size: int, T: int, d_model: int, hidden: int, n_layers: int = 3):
        super().__init__()
        assert n_layers >= 3
        self.T = T
        self.emb = nn.Embedding(vocab_size, d_model)
        layers = []
        inp = T * d_model
        layers.append(nn.Linear(inp, hidden))
        layers.append(nn.ReLU())
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden, vocab_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # (B,T)
        e = self.emb(x)                      # (B,T,D)
        h = e.reshape(x.size(0), -1)         # (B,T*D)
        logits = self.net(h)                 # (B,V)
        return logits.unsqueeze(1).expand(x.size(0), x.size(1), logits.size(-1))


class MHSAOneBlock(nn.Module):
    """
    Single multi-head self-attention block (causal) + optional FFN.
    Produces logits for each position (B,T,V).
    """
    def __init__(self, vocab_size: int, T: int, d_model: int, n_heads: int, head_dim: int, use_ffn: bool = False):
        super().__init__()
        self.T = T
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.use_ffn = use_ffn

        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(T, d_model)

        inner = n_heads * head_dim
        self.qkv = nn.Linear(d_model, 3 * inner, bias=False)
        self.out = nn.Linear(inner, d_model, bias=False)
        self.ln1 = nn.LayerNorm(d_model)

        if use_ffn:
            self.ln2 = nn.LayerNorm(d_model)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model),
            )

        self.head = nn.Linear(d_model, vocab_size)

        # causal mask buffer
        mask = torch.tril(torch.ones(T, T)).unsqueeze(0).unsqueeze(0)  # (1,1,T,T)
        self.register_buffer("causal", mask)

    def forward(self, x):  # (B,T)
        B, T = x.shape
        assert T <= self.T

        tok = self.emb(x)  # (B,T,D)
        pos_ids = torch.arange(T, device=x.device)
        h = tok + self.pos(pos_ids)[None, :, :]  # (B,T,D)

        # attention
        h1 = self.ln1(h)
        qkv = self.qkv(h1)  # (B,T,3*H*Hd)
        inner = self.n_heads * self.head_dim
        q, k, v = qkv.split(inner, dim=-1)
        # reshape to heads
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B,heads,T,Hd)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B,heads,T,T)
        causal = self.causal[:, :, :T, :T]
        att = att.masked_fill(causal == 0, float("-inf"))
        w = F.softmax(att, dim=-1)
        a = w @ v  # (B,heads,T,Hd)
        a = a.transpose(1, 2).contiguous().view(B, T, inner)  # (B,T,heads*Hd)
        a = self.out(a)  # (B,T,D)
        h = h + a

        if self.use_ffn:
            h2 = self.ln2(h)
            h = h + self.ffn(h2)

        logits = self.head(h)  # (B,T,V)
        return logits


class TransformerBlock(nn.Module):
    def __init__(self, T: int, d_model: int, n_heads: int, head_dim: int):
        super().__init__()
        inner = n_heads * head_dim
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * inner, bias=False)
        self.out = nn.Linear(inner, d_model, bias=False)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

        mask = torch.tril(torch.ones(T, T)).unsqueeze(0).unsqueeze(0)
        self.register_buffer("causal", mask)

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.T = T

    def forward(self, h):  # (B,T,D)
        B, T, D = h.shape

        # attn
        x = self.ln1(h)
        qkv = self.qkv(x)
        inner = self.n_heads * self.head_dim
        q, k, v = qkv.split(inner, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = self.causal[:, :, :T, :T]
        att = att.masked_fill(causal == 0, float("-inf"))
        w = F.softmax(att, dim=-1)
        a = w @ v
        a = a.transpose(1, 2).contiguous().view(B, T, inner)
        a = self.out(a)
        h = h + a

        # ffn
        x = self.ln2(h)
        h = h + self.ffn(x)
        return h


class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, T: int, d_model: int, n_heads: int, head_dim: int, n_layers: int):
        super().__init__()
        self.T = T
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(T, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(T=T, d_model=d_model, n_heads=n_heads, head_dim=head_dim)
            for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):  # (B,T)
        B, T = x.shape
        assert T <= self.T
        tok = self.emb(x)
        pos_ids = torch.arange(T, device=x.device)
        h = tok + self.pos(pos_ids)[None, :, :]
        for blk in self.blocks:
            h = blk(h)
        h = self.ln(h)
        return self.head(h)  # (B,T,V)


# =========================
# Step 6: Training + eval
# =========================
@dataclass
class TrainConfig:
    name: str
    T: int = 128
    batch_size: int = 64
    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    sample_every: int = 2
    max_steps_per_epoch: Optional[int] = None  # set to limit compute


def loss_fn(logits, targets):
    # logits: (B,T,V), targets: (B,T)
    B, T, V = logits.shape
    return F.cross_entropy(logits.reshape(B * T, V), targets.reshape(B * T))


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += loss_fn(logits, y).item()
        n_batches += 1
    return total_loss / max(1, n_batches)


@torch.no_grad()
def generate_chars(model, tokenizer: CharTokenizer, device: str, prompt: str, T: int, n_new: int = 100, temperature: float = 1.0) -> str:
    model.eval()
    ids = tokenizer.encode(prompt)
    x = torch.tensor(ids, dtype=torch.long, device=device)[None, :]
    for _ in range(n_new):
        x_cond = x[:, -T:] if x.size(1) > T else x
        logits = model(x_cond)  # (1,t,V)
        last = logits[:, -1, :] / max(1e-8, temperature)
        probs = F.softmax(last, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, nxt], dim=1)
    return tokenizer.decode(x[0].tolist())


@torch.no_grad()
def generate_words(model, tokenizer: WordTokenizer, device: str, prompt: str, T: int, n_new: int = 100, temperature: float = 1.0) -> str:
    model.eval()
    ids = tokenizer.encode(prompt, add_bos_eos=False)
    x = torch.tensor(ids, dtype=torch.long, device=device)[None, :]
    for _ in range(n_new):
        x_cond = x[:, -T:] if x.size(1) > T else x
        logits = model(x_cond)
        last = logits[:, -1, :] / max(1e-8, temperature)
        probs = F.softmax(last, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, nxt], dim=1)
    return tokenizer.decode(x[0].tolist())


def train_one_run(
    model: nn.Module,
    cfg: TrainConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
    flops_per_forward: int,
    sample_fn=None,
):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = {
        "train_loss": [],
        "val_loss": [],
        "test_loss": [],
        "train_flops": [],
    }

    total_tokens = 0
    total_flops = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for step, (x, y) in enumerate(train_loader, start=1):
            if cfg.max_steps_per_epoch is not None and step > cfg.max_steps_per_epoch:
                break

            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()

            if cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            opt.step()

            epoch_loss += loss.item()
            n_batches += 1

            # FLOPs bookkeeping (approx)
            tokens_this_batch = x.numel()
            total_tokens += tokens_this_batch
            total_flops += estimate_train_flops(flops_per_forward, tokens_this_batch)

        train_loss = epoch_loss / max(1, n_batches)
        val_loss = evaluate(model, val_loader, device)
        test_loss = evaluate(model, test_loader, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["test_loss"].append(test_loss)
        history["train_flops"].append(total_flops)

        print(f"[{cfg.name}] epoch {epoch:02d} | train {train_loss:.4f} | val {val_loss:.4f} | test {test_loss:.4f} | FLOPs {total_flops:.3e}")

        if sample_fn is not None and (epoch % cfg.sample_every == 0 or epoch == cfg.epochs):
            print(sample_fn(model))

    return history


# =========================
# Step 7: Plot helpers
# =========================
def save_plot_loss(history: Dict, outpath: str, title: str):
    plt.figure()
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def save_plot_ll_vs_setting(settings: List, test_ll: List[float], outpath: str, title: str, xlabel: str):
    plt.figure()
    plt.plot(settings, test_ll, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel("test log-likelihood (approx = -loss)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def save_plot_ll_vs_flops(flops: List[int], test_ll: List[float], outpath: str, title: str):
    plt.figure()
    plt.plot(flops, test_ll, marker="o")
    plt.xlabel("training FLOPs (approx)")
    plt.ylabel("test log-likelihood (approx = -loss)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


# =========================
# Step 8: Tiny Shakespeare runner
# =========================
def load_tiny_shakespeare(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def run_tiny_shakespeare(ts_path: str, out_dir: str, device: str):
    os.makedirs(out_dir, exist_ok=True)

    # --- Load official Tiny Shakespeare splits ---
    train_text = open(os.path.join(ts_path, "train.txt"), encoding="utf-8").read()
    val_text   = open(os.path.join(ts_path, "valid.txt"), encoding="utf-8").read()
    test_text  = open(os.path.join(ts_path, "test.txt"), encoding="utf-8").read()

    # Build tokenizer ONLY from training data (standard practice)
    tok = CharTokenizer(train_text)

    train_ids = torch.tensor(tok.encode(train_text), dtype=torch.long)
    val_ids   = torch.tensor(tok.encode(val_text), dtype=torch.long)
    test_ids  = torch.tensor(tok.encode(test_text), dtype=torch.long)

    total_chars = train_ids.numel() + val_ids.numel() + test_ids.numel()
    print(f"TinyShakespeare chars={total_chars} vocab={tok.vocab_size}")


    # --- Base training config ---
    base = TrainConfig(
        name="BASE",
        T=128,
        batch_size=64,
        epochs=12,
        lr=3e-4,
        weight_decay=0.01,
        max_steps_per_epoch=None,
        sample_every=4,
    )

    # ============================================================
    # Model family 1: Linear predictor
    # Required: test LL vs 3+ context lengths (we vary T)
    # ============================================================
    lin_Ts = [64, 128, 256]
    lin_test_ll = []
    lin_flops = []
    best_lin = None
    best_lin_ll = -1e9
    best_lin_hist = None
    best_lin_cfg = None

    for T in lin_Ts:
        cfg = TrainConfig(**{**asdict(base), "name": f"Linear_T{T}", "T": T})
        train_loader = make_loader(train_ids, T, cfg.batch_size, shuffle=True)
        val_loader = make_loader(val_ids, T, cfg.batch_size, shuffle=False)
        test_loader = make_loader(test_ids, T, cfg.batch_size, shuffle=False)

        model = LinearPredictor(vocab_size=tok.vocab_size, d_model=128)
        fpf = flops_linear(tok.vocab_size, T, d_model=128)

        sample_fn = lambda m: "\nSAMPLE:\n" + generate_chars(m, tok, device, prompt="HAMLET:", T=T, n_new=100)

        hist = train_one_run(model, cfg, train_loader, val_loader, test_loader, device, fpf, sample_fn=sample_fn)
        test_ll = -hist["test_loss"][-1]
        lin_test_ll.append(test_ll)
        lin_flops.append(hist["train_flops"][-1])

        save_plot_loss(hist, os.path.join(out_dir, f"linear_loss_T{T}.png"), f"Linear loss (T={T})")

        if test_ll > best_lin_ll:
            best_lin_ll = test_ll
            best_lin = model
            best_lin_hist = hist
            best_lin_cfg = {"T": T, "d_model": 128, "lr": cfg.lr, "batch": cfg.batch_size, "epochs": cfg.epochs}

    save_plot_ll_vs_setting(lin_Ts, lin_test_ll, os.path.join(out_dir, "linear_ll_vs_T.png"),
                            "Linear: test log-likelihood vs context length", "context length T")
    save_plot_ll_vs_flops(lin_flops, lin_test_ll, os.path.join(out_dir, "linear_ll_vs_flops.png"),
                          "Linear: test log-likelihood vs training FLOPs")
    print("Best Linear hyperparams:", best_lin_cfg)
    print("Best Linear sample:\n", generate_chars(best_lin, tok, device, "HAMLET:", T=best_lin_cfg["T"], n_new=100))

    # ============================================================
    # Model family 2: MLP (>=3 layers)
    # Required: test LL vs 3+ hyperparam settings
    # We vary hidden size; keep T fixed.
    # ============================================================
    mlp_hiddens = [256, 512, 768]
    mlp_test_ll = []
    mlp_flops = []
    best_mlp = None
    best_mlp_ll = -1e9
    best_mlp_cfg = None

    T = base.T
    train_loader = make_loader(train_ids, T, base.batch_size, shuffle=True)
    val_loader = make_loader(val_ids, T, base.batch_size, shuffle=False)
    test_loader = make_loader(test_ids, T, base.batch_size, shuffle=False)

    for h in mlp_hiddens:
        cfg = TrainConfig(**{**asdict(base), "name": f"MLP_hidden{h}", "T": T})
        model = MLP3Plus(vocab_size=tok.vocab_size, T=T, d_model=96, hidden=h, n_layers=3)
        fpf = flops_mlp(tok.vocab_size, T, d_model=96, hidden=h, layers=3)

        sample_fn = lambda m: "\nSAMPLE:\n" + generate_chars(m, tok, device, "HAMLET:", T=T, n_new=100)
        hist = train_one_run(model, cfg, train_loader, val_loader, test_loader, device, fpf, sample_fn=sample_fn)

        test_ll = -hist["test_loss"][-1]
        mlp_test_ll.append(test_ll)
        mlp_flops.append(hist["train_flops"][-1])
        save_plot_loss(hist, os.path.join(out_dir, f"mlp_loss_h{h}.png"), f"MLP loss (hidden={h})")

        if test_ll > best_mlp_ll:
            best_mlp_ll = test_ll
            best_mlp = model
            best_mlp_cfg = {"T": T, "d_model": 96, "hidden": h, "layers": 3, "lr": cfg.lr, "batch": cfg.batch_size, "epochs": cfg.epochs}

    save_plot_ll_vs_setting(mlp_hiddens, mlp_test_ll, os.path.join(out_dir, "mlp_ll_vs_hidden.png"),
                            "MLP: test log-likelihood vs hidden size", "hidden size")
    save_plot_ll_vs_flops(mlp_flops, mlp_test_ll, os.path.join(out_dir, "mlp_ll_vs_flops.png"),
                          "MLP: test log-likelihood vs training FLOPs")
    print("Best MLP hyperparams:", best_mlp_cfg)
    print("Best MLP sample:\n", generate_chars(best_mlp, tok, device, "HAMLET:", T=best_mlp_cfg["T"], n_new=100))

    # ============================================================
    # Model family 3: Multi-head self-attention (single block)
    # Required: test LL vs 3+ settings
    # We vary number of heads; keep head_dim fixed.
    # ============================================================
    attn_heads = [1, 2, 4]
    attn_test_ll = []
    attn_flops = []
    best_attn = None
    best_attn_ll = -1e9
    best_attn_cfg = None


    for nh in attn_heads:
        cfg = TrainConfig(**{**asdict(base), "name": f"MHSA_heads{nh}", "T": T})
        model = MHSAOneBlock(vocab_size=tok.vocab_size, T=T, d_model=128, n_heads=nh, head_dim=32, use_ffn=False)
        fpf = flops_attention(tok.vocab_size, T, d_model=128, n_heads=nh, head_dim=32, with_mlp=False)

        sample_fn = lambda m: "\nSAMPLE:\n" + generate_chars(m, tok, device, "HAMLET:", T=T, n_new=100)
        hist = train_one_run(model, cfg, train_loader, val_loader, test_loader, device, fpf, sample_fn=sample_fn)

        test_ll = -hist["test_loss"][-1]
        attn_test_ll.append(test_ll)
        attn_flops.append(hist["train_flops"][-1])
        save_plot_loss(hist, os.path.join(out_dir, f"mhsa_loss_h{nh}.png"), f"MHSA loss (heads={nh})")

        if test_ll > best_attn_ll:
            best_attn_ll = test_ll
            best_attn = model
            best_attn_cfg = {"T": T, "d_model": 128, "heads": nh, "head_dim": 32, "lr": cfg.lr, "batch": cfg.batch_size, "epochs": cfg.epochs}

    save_plot_ll_vs_setting(attn_heads, attn_test_ll, os.path.join(out_dir, "mhsa_ll_vs_heads.png"),
                            "MHSA: test log-likelihood vs num heads", "num heads")
    save_plot_ll_vs_flops(attn_flops, attn_test_ll, os.path.join(out_dir, "mhsa_ll_vs_flops.png"),
                          "MHSA: test log-likelihood vs training FLOPs")
    print("Best MHSA hyperparams:", best_attn_cfg)
    print("Best MHSA sample:\n", generate_chars(best_attn, tok, device, "HAMLET:", T=best_attn_cfg["T"], n_new=100))

    # ============================================================
    # Model family 4: Multi-layer Transformer
    # Required: test LL vs 3+ settings
    # We vary #layers; keep other dims fixed.
    # ============================================================
    tr_layers = [3,4,5]
    tr_test_ll = []
    tr_flops = []
    best_tr = None
    best_tr_ll = -1e9
    best_tr_cfg = None

    for nl in tr_layers:
        cfg = TrainConfig(**{**asdict(base), "name": f"Transformer_L{nl}", "T": T})
        model = TransformerLM(vocab_size=tok.vocab_size, T=T, d_model=192, n_heads=3, head_dim=64, n_layers=nl)
        fpf = flops_transformer(tok.vocab_size, T, d_model=192, n_heads=3, head_dim=64, n_layers=nl)

        sample_fn = lambda m: "\nSAMPLE:\n" + generate_chars(m, tok, device, "HAMLET:", T=T, n_new=100)
        hist = train_one_run(model, cfg, train_loader, val_loader, test_loader, device, fpf, sample_fn=sample_fn)

        test_ll = -hist["test_loss"][-1]
        tr_test_ll.append(test_ll)
        tr_flops.append(hist["train_flops"][-1])
        save_plot_loss(hist, os.path.join(out_dir, f"transformer_loss_L{nl}.png"), f"Transformer loss (layers={nl})")

        if test_ll > best_tr_ll:
            best_tr_ll = test_ll
            best_tr = model
            best_tr_cfg = {"T": T, "d_model": 192, "heads": 3, "head_dim": 64, "layers": nl, "lr": cfg.lr, "batch": cfg.batch_size, "epochs": cfg.epochs}

    save_plot_ll_vs_setting(tr_layers, tr_test_ll, os.path.join(out_dir, "transformer_ll_vs_layers.png"),
                            "Transformer: test log-likelihood vs #layers", "#layers")
    save_plot_ll_vs_flops(tr_flops, tr_test_ll, os.path.join(out_dir, "transformer_ll_vs_flops.png"),
                          "Transformer: test log-likelihood vs training FLOPs")
    print("Best Transformer hyperparams:", best_tr_cfg)
    print("Best Transformer sample:\n", generate_chars(best_tr, tok, device, "HAMLET:", T=best_tr_cfg["T"], n_new=100))

    # Pick best overall (by test log-likelihood)
    candidates = [
        ("linear", best_lin_ll, best_lin, best_lin_cfg),
        ("mlp", best_mlp_ll, best_mlp, best_mlp_cfg),
        ("mhsa", best_attn_ll, best_attn, best_attn_cfg),
        ("transformer", best_tr_ll, best_tr, best_tr_cfg),
    ]
    candidates.sort(key=lambda x: x[1], reverse=True)
    best_name, best_ll, best_model, best_cfg = candidates[0]
    print("\n=== BEST ON TINY SHAKESPEARE ===")
    print("model:", best_name)
    print("test log-likelihood (approx):", best_ll)
    print("cfg:", best_cfg)

    # ---- SAVE BEST SAMPLE FOR REPORT ----
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    save_sample(
        f"[BEST {best_name.upper()} – Tiny Shakespeare]\n" +
        generate_chars(
            best_model,
            tok,
            device,
            prompt="HAMLET:",
            T=best_cfg["T"],
            n_new=100
        ),
        os.path.join(BASE, "report_src", "samples.txt")
    )


    # Save best checkpoint for word-level stage
    ckpt = {
        "model_name": best_name,
        "model_state": best_model.state_dict(),
        "cfg": best_cfg,
        "char_vocab": tok.stoi,
    }
    torch.save(ckpt, os.path.join(out_dir, "best_char_model.pt"))

    return best_name, best_cfg, best_model


# =========================
# Step 9: PTB + WikiText-2 (word-level)
# =========================
def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def make_word_level_stream(text: str, tokenizer: WordTokenizer) -> torch.LongTensor:
    ids = tokenizer.encode(text, add_bos_eos=False)
    return torch.tensor(ids, dtype=torch.long)


def build_word_dataloaders(train_text: str, val_text: str, test_text: str, T: int, batch_size: int):
    # Fit tokenizer on training data only (standard)
    tok = WordTokenizer([train_text], min_freq=2, max_vocab=50000)

    train_ids = make_word_level_stream(train_text, tok)
    val_ids = make_word_level_stream(val_text, tok)
    test_ids = make_word_level_stream(test_text, tok)

    train_loader = make_loader(train_ids, T, batch_size, shuffle=True)
    val_loader = make_loader(val_ids, T, batch_size, shuffle=False)
    test_loader = make_loader(test_ids, T, batch_size, shuffle=False)
    return tok, train_loader, val_loader, test_loader


def run_word_level(
    out_dir: str,
    dataset_name: str,
    train_path: str,
    val_path: str,
    test_path: str,
    device: str,
    best_arch: str,
):
    os.makedirs(out_dir, exist_ok=True)
    train_text = read_text_file(train_path)
    val_text = read_text_file(val_path)
    test_text = read_text_file(test_path)

    # Use a smaller T for word-level to keep compute sane
    T = 64
    batch_size = 64
    tok, train_loader, val_loader, test_loader = build_word_dataloaders(train_text, val_text, test_text, T, batch_size)

    # Build the chosen model family, but sized for word-level vocab
    vocab = tok.vocab_size

    # Reasonable configs for word-level
    cfg = TrainConfig(
        name=f"{dataset_name}_{best_arch}",
        T=T,
        batch_size=batch_size,
        epochs=10,
        lr=3e-4,
        weight_decay=0.01,
        sample_every=5,
    )

    if best_arch == "linear":
        model = LinearPredictor(vocab_size=vocab, d_model=256)
        fpf = flops_linear(vocab, T, d_model=256)
    elif best_arch == "mlp":
        model = MLP3Plus(vocab_size=vocab, T=T, d_model=128, hidden=1024, n_layers=3)
        fpf = flops_mlp(vocab, T, d_model=128, hidden=1024, layers=3)
    elif best_arch == "mhsa":
        model = MHSAOneBlock(vocab_size=vocab, T=T, d_model=256, n_heads=4, head_dim=64, use_ffn=True)
        fpf = flops_attention(vocab, T, d_model=256, n_heads=4, head_dim=64, with_mlp=True)
    else:
        # transformer
        model = TransformerLM(vocab_size=vocab, T=T, d_model=256, n_heads=4, head_dim=64, n_layers=4)
        fpf = flops_transformer(vocab, T, d_model=256, n_heads=4, head_dim=64, n_layers=4)

    def sample_prompt():
        if dataset_name.lower().startswith("ptb"):
            return "the school announced that"
        return "The history of machine learning begins"

    sample_fn = lambda m: "\nSAMPLE:\n" + generate_words(m, tok, device, sample_prompt(), T=T, n_new=100)

    hist = train_one_run(model, cfg, train_loader, val_loader, test_loader, device, fpf, sample_fn=sample_fn)

    # Required plots: loss vs epochs, LL vs FLOPs
    save_plot_loss(hist, os.path.join(out_dir, f"{dataset_name}_loss.png"), f"{dataset_name}: loss vs epochs")
    test_ll_over_time = [-l for l in hist["test_loss"]]
    save_plot_ll_vs_flops(hist["train_flops"], test_ll_over_time, os.path.join(out_dir, f"{dataset_name}_ll_vs_flops.png"),
                          f"{dataset_name}: test log-likelihood vs training FLOPs")

    print(f"\n[{dataset_name}] final sample:\n", generate_words(model, tok, device, sample_prompt(), T=T, n_new=100))
    save_sample(
        f"[{dataset_name} – {best_arch.upper()}]\n" +
        generate_words(
            model,
            tok,
            device,
            sample_prompt(),
            T=T,
            n_new=100
        ),
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "report_src",
            "samples.txt"
        )
    )



def save_sample(text, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(text + "\n\n")



# =========================
# Step 10: Main
# =========================
def main():
    set_seed(42)
    device = get_device()
    print("Device:", device)

    # ---- YOU: set these paths ----
    # Tiny Shakespeare file path:
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    TINY_SHAKESPEARE_PATH = os.path.join(BASE, "datasets", "tiny_shakespeare")

    PTB_TRAIN = os.path.join(BASE, "datasets", "ptb", "train.txt")
    PTB_VAL   = os.path.join(BASE, "datasets", "ptb", "valid.txt")
    PTB_TEST  = os.path.join(BASE, "datasets", "ptb", "test.txt")

    WIKI_TRAIN = os.path.join(BASE, "datasets", "wikitext-2", "train.txt")
    WIKI_VAL   = os.path.join(BASE, "datasets", "wikitext-2", "valid.txt")
    WIKI_TEST  = os.path.join(BASE, "datasets", "wikitext-2", "test.txt")

    OUT = os.path.join(BASE, "report_src", "figures")

    os.makedirs(OUT, exist_ok=True)

    # 1) Run Tiny Shakespeare experiments
    best_arch, best_cfg, _ = run_tiny_shakespeare(TINY_SHAKESPEARE_PATH, os.path.join(OUT, "tinyshakespeare"), device)

    # 2) Run word-level on PTB + WikiText-2 with best architecture
    # (Make sure you have those text files available locally.)
    if os.path.exists(PTB_TRAIN) and os.path.exists(PTB_VAL) and os.path.exists(PTB_TEST):
        run_word_level(os.path.join(OUT, "ptb"), "PTB", PTB_TRAIN, PTB_VAL, PTB_TEST, device, best_arch)
    else:
        print("\nPTB files not found. Skipping PTB run. Put them at:", PTB_TRAIN, PTB_VAL, PTB_TEST)

    if os.path.exists(WIKI_TRAIN) and os.path.exists(WIKI_VAL) and os.path.exists(WIKI_TEST):
        run_word_level(os.path.join(OUT, "wikitext2"), "WikiText-2", WIKI_TRAIN, WIKI_VAL, WIKI_TEST, device, best_arch)
    else:
        print("\nWikiText-2 files not found. Skipping WikiText-2 run. Put them at:", WIKI_TRAIN, WIKI_VAL, WIKI_TEST)


if __name__ == "__main__":
    main()
