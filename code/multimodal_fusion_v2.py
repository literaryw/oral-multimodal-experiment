"""
多模态融合 v2 - 优化版
改进: 1) Low-Grade(G1+G2) vs High-Grade(G3+G4) 二分类
      2) Focal Loss 处理类别不平衡
      3) Cosine LR + 更多epoch + early stopping
      4) 更强的正则化
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, f1_score
from pathlib import Path
import copy

# Full HNSCC cohort (n=499) features. The canonical integrated/ directory has
# held only the strict oral subset (n=246) since 2026-08-10; the n=499 arrays
# used by this full-cohort script were restored from git history (commit
# 213aaa9) into integrated_full_legacy499/ so that this script truly runs on
# the full cohort it claims (n=499, Low=370/High=129).
DATA_DIR = Path(__file__).resolve().parent.parent / "data/TCGA-HNSC/integrated_full_legacy499"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print(f"Device: {DEVICE}")

# ===================== Focal Loss =====================

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # class weights
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        focal = (1 - pt) ** self.gamma * ce
        return focal.mean()

# ===================== Models =====================

class MLPBlock(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.BatchNorm1d(hidden), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.BatchNorm1d(hidden // 2), nn.Dropout(dropout),
            nn.Linear(hidden // 2, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class ImageOnly(nn.Module):
    def __init__(self, in_dim=1024, hidden=512, num_classes=2, dropout=0.4):
        super().__init__()
        self.encoder = MLPBlock(in_dim, hidden, hidden // 2, dropout)
        self.head = nn.Linear(hidden // 2, num_classes)

    def forward(self, x):
        h = self.encoder(x)
        return self.head(h), None

class TextOnly(nn.Module):
    def __init__(self, in_dim=768, hidden=512, num_classes=2, dropout=0.4):
        super().__init__()
        self.encoder = MLPBlock(in_dim, hidden, hidden // 2, dropout)
        self.head = nn.Linear(hidden // 2, num_classes)

    def forward(self, x):
        h = self.encoder(x)
        return self.head(h), None

class LateFusion(nn.Module):
    def __init__(self, img_dim=1024, text_dim=768, hidden=256, num_classes=2, dropout=0.4):
        super().__init__()
        self.img_enc = nn.Sequential(nn.Linear(img_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.text_enc = nn.Sequential(nn.Linear(text_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.BatchNorm1d(hidden), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes)
        )

    def forward(self, img, text):
        hi = self.img_enc(img)
        ht = self.text_enc(text)
        return self.head(torch.cat([hi, ht], -1)), None

class CrossAttentionFusion(nn.Module):
    def __init__(self, img_dim=1024, text_dim=768, hidden=256, num_classes=2, dropout=0.4, n_heads=4):
        super().__init__()
        self.img_proj = nn.Linear(img_dim, hidden)
        self.text_proj = nn.Linear(text_dim, hidden)
        self.cross_attn = nn.MultiheadAttention(hidden, n_heads, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.BatchNorm1d(hidden), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes)
        )

    def forward(self, img, text):
        hi = self.img_proj(img).unsqueeze(1)
        ht = self.text_proj(text).unsqueeze(1)
        attn_out, attn_w = self.cross_attn(hi, ht, ht)
        fused = torch.cat([hi.squeeze(1), attn_out.squeeze(1)], -1)
        return self.head(fused), attn_w

# ===================== Training =====================

def train_epoch(model, loader, opt, criterion, device, single_mod=False):
    model.train()
    loss_sum, correct, total = 0, 0, 0
    for batch in loader:
        if single_mod:
            x, y = batch[0].to(device), batch[1].to(device)
            opt.zero_grad()
            logits, _ = model(x)
        else:
            xi, xt, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            opt.zero_grad()
            logits, _ = model(xi, xt)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        loss_sum += loss.item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / len(loader), correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device, single_mod=False):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for batch in loader:
        if single_mod:
            x, y = batch[0].to(device), batch[1].to(device)
            logits, _ = model(x)
        else:
            xi, xt, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            logits, _ = model(xi, xt)
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(y.cpu().tolist())
        all_probs.extend(torch.softmax(logits, 1).cpu().tolist())
    return all_preds, all_labels, all_probs

# ===================== Main =====================

def run_cv(name, model_cls, single_mod, X_list, y, n_splits=5):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    all_preds, all_labels, all_probs = [], [], []

    for fold, (tr, val) in enumerate(kf.split(X_list[0], y)):
        # Data
        if single_mod:
            Xt, yt = X_list[0][tr], y[tr]
            Xv, yv = X_list[0][val], y[val]
            # Weighted sampler for train
            class_counts = np.bincount(yt)
            weights = 1.0 / class_counts[yt]
            sampler = WeightedRandomSampler(torch.DoubleTensor(weights), len(yt), replacement=True)
            tr_ds = TensorDataset(torch.FloatTensor(Xt), torch.LongTensor(yt))
            tr_loader = DataLoader(tr_ds, batch_size=32, sampler=sampler)
            val_ds = TensorDataset(torch.FloatTensor(Xv), torch.LongTensor(yv))
            val_loader = DataLoader(val_ds, batch_size=64)
        else:
            Xi_t, Xt_t = X_list[0][tr], X_list[1][tr]
            Xi_v, Xt_v = X_list[0][val], X_list[1][val]
            yt, yv = y[tr], y[val]
            class_counts = np.bincount(yt)
            weights = 1.0 / class_counts[yt]
            sampler = WeightedRandomSampler(torch.DoubleTensor(weights), len(yt), replacement=True)
            tr_ds = TensorDataset(torch.FloatTensor(Xi_t), torch.FloatTensor(Xt_t), torch.LongTensor(yt))
            tr_loader = DataLoader(tr_ds, batch_size=32, sampler=sampler)
            val_ds = TensorDataset(torch.FloatTensor(Xi_v), torch.FloatTensor(Xt_v), torch.LongTensor(yv))
            val_loader = DataLoader(val_ds, batch_size=64)

        # Model
        model = model_cls().to(DEVICE)
        # Focal loss with class-specific alpha
        alpha = torch.FloatTensor([1.0, len(yt) / class_counts.max()]).to(DEVICE)
        criterion = FocalLoss(alpha=alpha, gamma=2.0)
        opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        scheduler = CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2)

        best_model = None
        best_f1 = 0
        patience = 15
        no_improve = 0

        for epoch in range(100):
            train_epoch(model, tr_loader, opt, criterion, DEVICE, single_mod)
            preds, labels, probs = evaluate(model, val_loader, criterion, DEVICE, single_mod)
            scheduler.step()
            val_f1 = f1_score(labels, preds, average='macro')

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_model = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        # Load best model and evaluate
        model.load_state_dict(best_model)
        preds, labels, probs = evaluate(model, val_loader, criterion, DEVICE, single_mod)
        all_preds.extend(preds); all_labels.extend(labels); all_probs.extend(probs)

        acc = sum(1 for p, l in zip(preds, labels) if p == l) / len(labels)
        print(f"  Fold {fold+1}: Acc={acc:.3f} F1={f1_score(labels, preds, average='macro'):.3f} (best={best_f1:.3f}, epochs={epoch-no_improve})")

    return all_preds, all_labels, all_probs

if __name__ == "__main__":
    # Load and prepare data
    X_img = np.load(DATA_DIR / "X_img_uni.npy")
    X_text = np.load(DATA_DIR / "X_text_biomedbert.npy")
    y_4class = np.load(DATA_DIR / "y.npy")

    # Binary: 0=Low-Grade(G1+G2), 1=High-Grade(G3+G4)
    y_binary = np.where(np.isin(y_4class, [0, 1]), 0, 1)  # 0,1->low, 2,3->high
    n_low = (y_binary == 0).sum()
    n_high = (y_binary == 1).sum()
    print(f"Binary labels: Low-grade={n_low}, High-grade={n_high}")

    experiments = [
        ("Image-Only (UNI)",              ImageOnly, ['img']),
        ("Text-Only (BiomedBERT)",         TextOnly,  ['txt']),
        ("LateFusion (UNI+BiomedBERT)",    LateFusion, ['img', 'txt']),
        ("CrossAttention (UNI+BiomedBERT)", CrossAttentionFusion, ['img', 'txt']),
    ]

    all_runs = {}
    for name, model_cls, modalities in experiments:
        # Reset RNG per model to ensure execution-order independence (protocol-consistent)
        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        print(f"\n{'='*55}")
        print(f"  {name}")
        print(f"{'='*55}")
        if modalities == ['img']:
            X_list, single = [X_img], True
        elif modalities == ['txt']:
            X_list, single = [X_text], True
        else:
            X_list, single = [X_img, X_text], False
        preds, labels, probs = run_cv(name, model_cls, single, X_list, y_binary)
        all_runs[name] = (preds, labels, probs)

        acc = sum(1 for p, l in zip(preds, labels) if p == l) / len(labels)
        print(f"\n  Results:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1 (macro): {f1_score(labels, preds, average='macro'):.4f}")
        print(classification_report(labels, preds, target_names=['Low-Grade', 'High-Grade'], zero_division=0))

        try:
            auc = roc_auc_score(labels, [p[1] for p in probs])
            print(f"  AUC: {auc:.4f}")
        except:
            pass

        cm = confusion_matrix(labels, preds)
        print(f"  Confusion Matrix:\n   {cm}")

    # ====== Save protocol-consistent results (Table 3 full-column source) ======
    import json
    out_dir = Path(__file__).resolve().parent.parent / "data" / "TCGA-HNSC" / "experiment_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = {"results": {}, "confusion_matrices": {},
             "metadata": {"dataset": "TCGA-HNSC (n=499, full cohort)",
                          "cv": "5-fold stratified",
                          "protocol": "per-model RNG reset (seed=42)"}}
    for name, (preds, labels, probs) in all_runs.items():
        preds_arr, labels_arr = np.array(preds), np.array(labels)
        rep = classification_report(labels_arr, preds_arr,
                                    target_names=['Low-Grade', 'High-Grade'],
                                    zero_division=0, output_dict=True)
        acc = float((preds_arr == labels_arr).mean())
        macro_f1 = float(f1_score(labels_arr, preds_arr, average='macro'))
        try:
            auc_v = float(roc_auc_score(labels_arr, [p[1] for p in probs]))
        except ValueError:
            auc_v = 0.5
        saved["results"][name] = {
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "auc": round(auc_v, 4),
            "low_f1": round(float(rep['Low-Grade']['f1-score']), 4),
            "high_f1": round(float(rep['High-Grade']['f1-score']), 4),
        }
        saved["confusion_matrices"][name] = confusion_matrix(labels_arr, preds_arr).tolist()
    json.dump(saved, open(out_dir / "main_experiment_results.json", "w"), indent=2)
    print(f"\nSaved main_experiment_results.json (protocol-consistent, full cohort)")

    print("\n=== DONE ===")
