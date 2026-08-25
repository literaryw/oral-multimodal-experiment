"""
Roadmap 改进实验: Modality Dropout — 随机屏蔽一个模态, 强制独立学习
"""
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score
from pathlib import Path
import copy

DATA_DIR = Path("../data/TCGA-HNSC/integrated")
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

class ModalityDropoutFusion(nn.Module):
    """Cross-attention fusion with modality dropout"""
    def __init__(self, img_dim=1024, text_dim=768, hidden=256, num_classes=2, drop_rate=0.3):
        super().__init__()
        self.drop_rate = drop_rate
        self.img_proj = nn.Sequential(nn.Linear(img_dim, hidden), nn.ReLU(), nn.Dropout(0.4))
        self.text_proj = nn.Sequential(nn.Linear(text_dim, hidden), nn.ReLU(), nn.Dropout(0.4))
        self.cross_attn = nn.MultiheadAttention(hidden, 4, batch_first=True, dropout=0.2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden*2, hidden), nn.ReLU(), nn.BatchNorm1d(hidden), nn.Dropout(0.4),
            nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(hidden//2, num_classes))

    def forward(self, img, text, training=True):
        if training and torch.rand(1).item() < self.drop_rate:
            # Drop image modality -> use only text
            img = torch.zeros_like(img)
        elif training and torch.rand(1).item() < self.drop_rate:
            # Drop text modality -> use only image
            text = torch.zeros_like(text)

        hi = self.img_proj(img).unsqueeze(1)
        ht = self.text_proj(text).unsqueeze(1)
        attn_out, _ = self.cross_attn(hi, ht, ht)
        fused = torch.cat([hi.squeeze(1), attn_out.squeeze(1)], -1)
        return self.classifier(fused), None

class CrossAttentionFusion(nn.Module):
    """Baseline: cross-attention without dropout"""
    def __init__(self, img_dim=1024, text_dim=768, hidden=256, num_classes=2):
        super().__init__()
        self.img_proj = nn.Sequential(nn.Linear(img_dim, hidden), nn.ReLU(), nn.Dropout(0.4))
        self.text_proj = nn.Sequential(nn.Linear(text_dim, hidden), nn.ReLU(), nn.Dropout(0.4))
        self.cross_attn = nn.MultiheadAttention(hidden, 4, batch_first=True, dropout=0.2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden*2, hidden), nn.ReLU(), nn.BatchNorm1d(hidden), nn.Dropout(0.4),
            nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(hidden//2, num_classes))

    def forward(self, img, text, training=True):
        hi = self.img_proj(img).unsqueeze(1)
        ht = self.text_proj(text).unsqueeze(1)
        attn_out, _ = self.cross_attn(hi, ht, ht)
        fused = torch.cat([hi.squeeze(1), attn_out.squeeze(1)], -1)
        return self.classifier(fused), None

def train_epoch(model, loader, opt, crit, device):
    model.train()
    L_t, C_t, T_t = 0, 0, 0
    for xi, xt, y in loader:
        xi, xt, y = xi.to(device), xt.to(device), y.to(device)
        opt.zero_grad()
        logits, _ = model(xi, xt, training=True)
        loss = crit(logits, y); loss.backward()
        opt.step()
        L_t+=loss.item(); C_t+=(logits.argmax(1)==y).sum().item(); T_t+=y.size(0)
    return L_t/len(loader), C_t/T_t

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    P, L, R = [], [], []
    for xi, xt, y in loader:
        xi, xt, y = xi.to(device), xt.to(device), y.to(device)
        logits, _ = model(xi, xt, training=False)
        P.extend(logits.argmax(1).cpu()); L.extend(y.cpu())
        R.extend(torch.softmax(logits,1).cpu().tolist())
    return P, L, R

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    Xi = np.load(DATA_DIR / "X_img_uni.npy")
    Xt = np.load(DATA_DIR / "X_text_biomedbert.npy")
    y4 = np.load(DATA_DIR / "y.npy")
    y = np.where(np.isin(y4, [0,1]), 0, 1)
    print(f"Cases: {len(y)} (Low={sum(y==0)}, High={sum(y==1)})")

    kf = StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED)

    for name, cls in [("CrossAttn (baseline)", CrossAttentionFusion),
                       ("CrossAttn + ModDrop", ModalityDropoutFusion)]:
        P, L, R = [], [], []
        for tr, val in kf.split(Xi, y):
            Xit, Xiv = Xi[tr], Xi[val]; Xtt, Xtv = Xt[tr], Xt[val]
            yt, yv = y[tr], y[val]

            tr_ds = TensorDataset(torch.FloatTensor(Xit), torch.FloatTensor(Xtt), torch.LongTensor(yt))
            v_ds = TensorDataset(torch.FloatTensor(Xiv), torch.FloatTensor(Xtv), torch.LongTensor(yv))
            tr_ldr = DataLoader(tr_ds, 32, True); v_ldr = DataLoader(v_ds, 64)

            model = cls().to(device)
            crit = nn.CrossEntropyLoss(weight=torch.FloatTensor([1.0, 3.0]).to(device))
            opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

            for ep in range(80): train_epoch(model, tr_ldr, opt, crit, device)
            p, l, r = evaluate(model, v_ldr, device)
            P.extend(p); L.extend(l); R.extend(r)

        acc = sum(1 for a,b in zip(P,L) if a==b)/len(L)
        from sklearn.metrics import f1_score
        f1 = f1_score(L, P, average='macro')
        print(f"\n{name}: Acc={acc:.3f} F1={f1:.3f}")
        print(classification_report(L, P, target_names=['Low','High'], zero_division=0))
        try:
            auc = roc_auc_score(L, [r[1] for r in R])
            print(f"AUC={auc:.3f}")
        except: pass
