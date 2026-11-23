"""Train the auxiliary text proxy and archive its latent-chunk attention.

The model operates on four chunks created after a learned dense projection of the
768-dimensional embedding.  The chunks have no validated clinical-field meaning.
"""
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
import copy

_PROJ = Path(__file__).resolve().parent.parent
DATA_DIR = _PROJ / "data/TCGA-HNSC/integrated_full_legacy499"
OUT_DIR = _PROJ / "data/TCGA-HNSC/experiment_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = _PROJ / "paper"
RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)

class TextClassifierWithAttention(nn.Module):
    """
    Auxiliary classifier with self-attention over four latent chunks.
    """
    def __init__(self, in_dim=768, hidden=512, num_classes=2, dropout=0.4, n_heads=4):
        super().__init__()
        # A learned dense projection is divided into four structural chunks.
        self.token_dim = in_dim // 4  # 192
        self.n_tokens = 4
        self.token_proj = nn.Linear(in_dim, in_dim)

        # Multi-head self-attention on token_dim
        self.attention = nn.MultiheadAttention(self.token_dim, n_heads, batch_first=True, dropout=0.2)

        # Project attention output back to original dim
        self.out_proj = nn.Linear(self.token_dim, in_dim)

        # MLP classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.BatchNorm1d(hidden), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.BatchNorm1d(hidden // 2), nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes)
        )
        self.saved_attn_weights = None

    def forward(self, x):
        B = x.size(0)
        # Divide the learned projection into four contiguous latent chunks.
        h = self.token_proj(x)  # (B, 768)
        tokens = h.view(B, self.n_tokens, self.token_dim)  # (B, 4, 192)

        # Self-attention over four latent chunks.
        attn_out, attn_weights = self.attention(tokens, tokens, tokens)
        # attn_out: (B, 4, 192), attn_weights: (B, 4, 4)
        self.saved_attn_weights = attn_weights.detach()

        # 聚合: 平均池化 + 投影回 768-dim
        pooled = attn_out.mean(dim=1)  # (B, 192)
        pooled = self.out_proj(pooled)  # (B, 768)
        return self.classifier(pooled)

def train_and_extract_attention():
    print(f"Device: {DEVICE}")
    print("Training BiomedBERT classifier and extracting REAL attention weights...\n")

    # 加载数据
    X = np.load(DATA_DIR / "X_text_biomedbert.npy")
    y4 = np.load(DATA_DIR / "y.npy")
    if X.shape != (499, 768) or y4.shape[0] != 499:
        raise RuntimeError("The archived auxiliary proxy requires the retained ordered 499-case inputs")
    y = np.where(np.isin(y4, [0, 1]), 0, 1)

    # 5-fold CV
    kf = StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED)
    all_attn_weights = []

    for fold, (tr, val) in enumerate(kf.split(X, y)):
        Xt, yt = X[tr], y[tr]

        class_counts = np.bincount(yt)
        weights_sampler = 1.0 / class_counts[yt]
        from torch.utils.data import WeightedRandomSampler
        sampler = WeightedRandomSampler(torch.DoubleTensor(weights_sampler), len(yt), replacement=True)
        tr_ds = TensorDataset(torch.FloatTensor(Xt), torch.LongTensor(yt))
        tr_loader = DataLoader(tr_ds, batch_size=32, sampler=sampler)

        model = TextClassifierWithAttention().to(DEVICE)
        alpha = torch.FloatTensor([1.0, len(yt) / class_counts.max()]).to(DEVICE)

        from multimodal_fusion_v2 import FocalLoss
        criterion = FocalLoss(alpha=alpha, gamma=2.0)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

        for ep in range(50):
            model.train()
            for xb, yb in tr_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                opt.step()

        # 提取整个验证集的注意力权重
        Xv, yv = X[val], y[val]
        model.eval()
        fold_attns = []
        with torch.no_grad():
            for i in range(0, len(Xv), 64):
                xb = torch.FloatTensor(Xv[i:i+64]).to(DEVICE)
                _ = model(xb)
                fold_attns.append(model.saved_attn_weights.cpu().numpy())

        all_attn_weights.append(np.concatenate(fold_attns, axis=0))
        print(f"  Fold {fold+1}: extracted attention for {len(Xv)} samples")

    # 合并所有 fold 的注意力权重
    all_attn = np.concatenate(all_attn_weights, axis=0)  # (499, 4, 4)

    # Aggregate raw attention without assigning clinical semantics to chunks.
    mean_attn = all_attn.mean(axis=0)  # (4, 4)
    avg_per_key = mean_attn.mean(axis=0)  # (4,) — 每个 key token 被关注的程度

    clean_archive = OUT_DIR / "text_latent_attention_archive.npz"
    np.savez_compressed(
        clean_archive,
        raw_attn=all_attn.astype(np.float32),
        avg_per_key=avg_per_key.astype(np.float32),
        chunk_width=np.int64(192),
        definition=np.array(
            "Four contiguous chunks after learned dense projection; no clinical-field mapping"
        ),
    )

    from fig2_regenerate import render_latent_chunk_attention

    render_latent_chunk_attention(clean_archive, FIG_DIR / "fig2_text_weights.pdf")
    print("\n  Latent-chunk attention archive and manuscript figure saved")

    # 保存模型
    torch.save(model.state_dict(), OUT_DIR / "text_attention_model.pt")

    return avg_per_key

if __name__ == "__main__":
    train_and_extract_attention()
