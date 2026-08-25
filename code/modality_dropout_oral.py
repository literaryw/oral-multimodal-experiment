"""
口腔亚组模态缺失消融 (Modality Dropout on n=246)
验证 modality dropout 能否在保持准确率的同时救回 High-Grade 召回
"""
import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

# Import base utilities from oral_experiments_full
with open("oral_experiments_full.py", encoding="utf-8") as f:
    src = f.read()
top_src = src.split("if __name__")[0]
exec(top_src)

from pathlib import Path
import json

BASE_DIR = Path(".").resolve().parent
d = BASE_DIR / "data" / "TCGA-HNSC" / "integrated"
OUTPUT_DIR = BASE_DIR / "data" / "TCGA-HNSC" / "experiment_outputs"

X_img = np.load(d / "X_img_uni.npy")
X_text = np.load(d / "X_text_biomedbert.npy")
y_4class = np.load(d / "y.npy")
y_binary = np.where(np.isin(y_4class, [0, 1]), 0, 1)
print(f"Data: {len(y_binary)} cases, Low={int((y_binary==0).sum())}, High={int((y_binary==1).sum())}")

# ====== Modality Dropout Model ======
class ModalityDropoutFusion(nn.Module):
    """Cross-attention fusion with per-sample modality dropout during training"""
    def __init__(self, img_dim=1024, text_dim=768, hidden=256, num_classes=2,
                 dropout=0.4, drop_rate=0.3, n_heads=4):
        super().__init__()
        self.drop_rate = drop_rate
        self.img_proj = nn.Linear(img_dim, hidden)
        self.text_proj = nn.Linear(text_dim, hidden)
        self.cross_attn = nn.MultiheadAttention(hidden, n_heads, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden*2, hidden), nn.ReLU(), nn.BatchNorm1d(hidden), nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden//2, num_classes))

    def forward(self, img, text, training=True):
        if training:
            # Per-sample modality dropout (not batch-level coin flip)
            batch_size = img.size(0)
            drop_img = torch.rand(batch_size, 1) < self.drop_rate
            drop_text = torch.rand(batch_size, 1) < self.drop_rate
            img = img.clone()
            text = text.clone()
            img[drop_img.expand_as(img)] = 0.0
            text[drop_text.expand_as(text)] = 0.0
        hi, ht = self.img_proj(img).unsqueeze(1), self.text_proj(text).unsqueeze(1)
        attn_out, _ = self.cross_attn(hi, ht, ht)
        return self.head(torch.cat([hi.squeeze(1), attn_out.squeeze(1)], -1)), _

# ====== Modified training with modality dropout ======
def train_epoch_moddrop(model, loader, opt, criterion, device):
    model.train()
    for batch in loader:
        xi, xt, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        opt.zero_grad()
        logits, _ = model(xi, xt, training=True)
        criterion(logits, y).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

@torch.no_grad()
def evaluate_moddrop(model, loader, device):
    model.eval()
    preds, labels, probs = [], [], []
    for batch in loader:
        xi, xt, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        logits, _ = model(xi, xt, training=False)
        preds.extend(logits.argmax(1).cpu().tolist())
        labels.extend(y.cpu().tolist())
        probs.extend(torch.softmax(logits, 1).cpu().tolist())
    return preds, labels, probs

def run_moddrop_cv(name, model_factory, X_list, y, n_splits=5):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    fold_results = []
    for fold, (tr, val) in enumerate(kf.split(X_list[0], y)):
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

        model = model_factory().to(DEVICE)
        alpha = torch.FloatTensor([1.0, len(yt)/class_counts.max()]).to(DEVICE)
        criterion = FocalLoss(alpha=alpha, gamma=2.0)
        opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        scheduler = CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2)

        best_f1, best_state, no_improve = 0, None, 0
        for epoch in range(100):
            train_epoch_moddrop(model, tr_loader, opt, criterion, DEVICE)
            preds, labels, probs = evaluate_moddrop(model, val_loader, DEVICE)
            scheduler.step()
            val_f1 = f1_score(labels, preds, average='macro')
            if val_f1 > best_f1:
                best_f1, best_state = val_f1, copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= 15:
                    break

        model.load_state_dict(best_state)
        preds, labels, probs = evaluate_moddrop(model, val_loader, DEVICE)
        cm = confusion_matrix(labels, preds)
        res = {
            "accuracy": float(accuracy_score(labels, preds)),
            "macro_f1": float(f1_score(labels, preds, average='macro')),
            "low_f1": float(f1_score(labels, preds, average=None)[0]) if len(np.unique(labels)) > 1 else 0.0,
            "high_f1": float(f1_score(labels, preds, average=None)[1]) if len(np.unique(labels)) > 1 else 0.0,
            "auc": float(roc_auc_score(labels, [p[1] for p in probs])),
            "cm": cm.tolist(),
        }
        fold_results.append(res)
        print(f"  Fold {fold+1}: Acc={res['accuracy']:.3f} F1={res['macro_f1']:.3f} "
              f"AUC={res['auc']:.3f} HighF1={res['high_f1']:.3f}")

    keys = ["accuracy", "macro_f1", "low_f1", "high_f1", "auc"]
    agg = {}
    for k in keys:
        vals = [f[k] for f in fold_results]
        ci = stats.t.interval(0.95, len(vals)-1, loc=np.mean(vals), scale=np.std(vals, ddof=1)/np.sqrt(len(vals)))
        agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)),
                  "ci95_low": float(ci[0]), "ci95_high": float(ci[1]), "per_fold": vals}
    return {"aggregate": agg, "per_fold": fold_results}

# ====== Run comparison ======
# Reset RNG per model/condition to ensure execution-order independence (protocol-consistent)
print("\n=== Cross-Attention Baseline (no dropout) ===")
torch.manual_seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)
ca_result = run_cv_with_stats("CrossAttn Baseline",
    lambda: CrossAttentionFusion(), False, [X_img, X_text], y_binary, 5)

print("\n=== Cross-Attention + Modality Dropout (p=0.3) ===")
torch.manual_seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)
md_result = run_moddrop_cv("CrossAttn + ModDrop",
    lambda: ModalityDropoutFusion(drop_rate=0.3), [X_img, X_text], y_binary, 5)

ca_agg = ca_result["aggregate"]
md_agg = md_result["aggregate"]

print(f"\n=== Comparison ===")
print(f"  Baseline:   Acc={ca_agg['accuracy']['mean']:.3f} F1={ca_agg['macro_f1']['mean']:.3f} AUC={ca_agg['auc']['mean']:.3f} HighF1={ca_agg['high_f1']['mean']:.3f}")
print(f"  +ModDrop:   Acc={md_agg['accuracy']['mean']:.3f} F1={md_agg['macro_f1']['mean']:.3f} AUC={md_agg['auc']['mean']:.3f} HighF1={md_agg['high_f1']['mean']:.3f}")
delta_highf1 = md_agg['high_f1']['mean'] - ca_agg['high_f1']['mean']
delta_auc = md_agg['auc']['mean'] - ca_agg['auc']['mean']
print(f"  Delta HighF1: {delta_highf1:+.3f}, Delta AUC: {delta_auc:+.3f}")

results = {
    "Cross-Attention (baseline)": {"aggregate": ca_agg, "per_fold": ca_result["per_fold"]},
    "Cross-Attention + Modality Dropout (p=0.3)": {"aggregate": md_agg, "per_fold": md_result["per_fold"]},
    "_interpretation": {
        "high_f1_change": float(delta_highf1),
        "auc_change": float(delta_auc),
        "verdict": (
            f"Modality dropout {'IMPROVES' if delta_highf1 > 0.05 else 'marginally affects'} "
            f"High-Grade recall (delta HighF1={delta_highf1:+.3f}) on the strict oral subset. "
            f"{'ModDrop partially rescues minority-class discrimination.' if delta_highf1 > 0.05 else 'ModDrop alone is insufficient to overcome extreme class imbalance (49 High-Grade cases).'}"
        ),
    },
}
json.dump(results, open(OUTPUT_DIR / "oral_moddrop_results.json", "w"), indent=2)
print(f"\nSaved oral_moddrop_results.json")
print("DONE")
