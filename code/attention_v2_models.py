"""
R16-fix Phase B: model definitions for the true-cross-attention re-run.

`SingleTokenAttentionFusion` is the degenerate head, under its final name.  The module
has gone through two misleading names: it was originally called "CrossAttentionFusion"
(though it fed `nn.MultiheadAttention` sequences of length 1 via `.unsqueeze(1)`, so the
softmax over the single key is identically 1.0 and the module reduces to
`concat(W_i @ img, Linear(text))` with no image-text interaction), and an intermediate
revision called it "gated-bilinear" although it has neither a gate nor a bilinear
product.  Both earlier names remain importable as aliases so that archived outputs and
the `gated_bilinear` arm key stay valid; the class itself is named after the operation
it performs.  It is kept as a documented failure mode.

`TrueCrossAttentionFusion` is the genuine article: patch tokens form the query
sequence (N per case, 1..88, median 13) and text tokens form the key/value sequence
(L per case, 17..48, median 37), with padding masks on both sides.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        return ((1 - torch.exp(-ce)) ** self.gamma * ce).mean()


class MLPBlock(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.BatchNorm1d(hidden), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.BatchNorm1d(hidden // 2), nn.Dropout(dropout),
            nn.Linear(hidden // 2, out_dim))

    def forward(self, x):
        return self.net(x)


class ImageOnly(nn.Module):
    def __init__(self, in_dim=1024, hidden=512, num_classes=2, dropout=0.4):
        super().__init__()
        self.encoder = MLPBlock(in_dim, hidden, hidden // 2, dropout)
        self.head = nn.Linear(hidden // 2, num_classes)

    def forward(self, x):
        return self.head(self.encoder(x)), None


class TextOnly(nn.Module):
    def __init__(self, in_dim=768, hidden=512, num_classes=2, dropout=0.4):
        super().__init__()
        self.encoder = MLPBlock(in_dim, hidden, hidden // 2, dropout)
        self.head = nn.Linear(hidden // 2, num_classes)

    def forward(self, x):
        return self.head(self.encoder(x)), None


class LateFusion(nn.Module):
    def __init__(self, img_dim=1024, text_dim=768, hidden=256, num_classes=2, dropout=0.4):
        super().__init__()
        self.img_enc = nn.Sequential(nn.Linear(img_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.text_enc = nn.Sequential(nn.Linear(text_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.BatchNorm1d(hidden), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes))

    def forward(self, img, text):
        return self.head(torch.cat([self.img_enc(img), self.text_enc(text)], -1)), None


class SingleTokenAttentionFusion(nn.Module):
    """Projection + single-token attention + concatenation head.

    Formerly `CrossAttentionFusion`, then `GatedBilinearFusion`; both earlier names are
    misleading (see the module docstring).  `unsqueeze(1)` makes the sequence length 1 on
    both sides, so the softmax over the single key is always 1.0 and the attention output
    is a plain linear map of the text vector, independent of the image query.
    Numerically identical to the module that produced the published cross-attention rows.
    """

    def __init__(self, img_dim=1024, text_dim=768, hidden=256, num_classes=2, dropout=0.4, n_heads=4):
        super().__init__()
        self.img_proj = nn.Linear(img_dim, hidden)
        self.text_proj = nn.Linear(text_dim, hidden)
        self.cross_attn = nn.MultiheadAttention(hidden, n_heads, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.BatchNorm1d(hidden), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes))

    def forward(self, img, text):
        hi, ht = self.img_proj(img).unsqueeze(1), self.text_proj(text).unsqueeze(1)
        attn_out, _ = self.cross_attn(hi, ht, ht)
        return self.head(torch.cat([hi.squeeze(1), attn_out.squeeze(1)], -1)), _


# Back-compat aliases: older scripts and archived outputs refer to these names.
GatedBilinearFusion = SingleTokenAttentionFusion
CrossAttentionFusion = SingleTokenAttentionFusion


class TrueCrossAttentionFusion(nn.Module):
    """Genuine cross-modal attention: patch tokens query text tokens.

    Q = image patch tokens (N, 1024), K = V = clinical text tokens (L, 768).
    Both sequences carry padding masks; the attended patch representations are
    mean-pooled over real patches only, then concatenated with the masked-mean of
    the text tokens (NOT the [CLS] token) and classified.  This is the module the
    manuscript's Eq. (3) describes (`Q = X_I, K = V = X_T`).
    """

    def __init__(self, img_dim=1024, text_dim=768, hidden=256, num_classes=2,
                 dropout=0.4, n_heads=4):
        super().__init__()
        self.img_proj = nn.Linear(img_dim, hidden)
        self.text_proj = nn.Linear(text_dim, hidden)
        self.cross_attn = nn.MultiheadAttention(hidden, n_heads, batch_first=True, dropout=dropout)
        self.norm_q = nn.LayerNorm(hidden)
        self.norm_t = nn.LayerNorm(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.BatchNorm1d(hidden), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes))

    def forward(self, patches, patch_mask, text_tokens, text_mask):
        """patches (B,N,1024); patch_mask (B,N) 1=real; text_tokens (B,L,768); text_mask (B,L) 1=real."""
        q = self.norm_q(self.img_proj(patches))          # (B, N, H)
        kv = self.norm_t(self.text_proj(text_tokens))    # (B, L, H)

        attn_out, attn_w = self.cross_attn(
            q, kv, kv, key_padding_mask=(text_mask == 0), need_weights=True)

        # mean-pool attended patch representations over real patches only
        m = patch_mask.unsqueeze(-1).to(attn_out.dtype)  # (B, N, 1)
        pooled = (attn_out * m).sum(1) / m.sum(1).clamp(min=1.0)

        # text summary = masked mean of text tokens
        mt = text_mask.unsqueeze(-1).to(kv.dtype)
        text_summary = (kv * mt).sum(1) / mt.sum(1).clamp(min=1.0)

        return self.head(torch.cat([pooled, text_summary], -1)), attn_w
