"""
文本单模态基线: 用BiomedBERT特征做OSCC分级
评估MLP vs 树状决策层的性能差异
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

FEATURE_DIR = Path("../data/TCGA-HNSC/features")
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ===================== 模型定义 =====================

class MLPClassifier(nn.Module):
    """简单MLP基线"""
    def __init__(self, input_dim=768, hidden_dim=256, num_classes=4, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.net(x)

class TreeStructuredClassifier(nn.Module):
    """树状决策层: 先检测良恶,再细分级别
    Node 0: G1/G2 vs G3/G4 (低级别 vs 高级别)
    Node 1: G1 vs G2 (高分化 vs 中分化)
    Node 2: G3 vs G4 (低分化 vs 未分化)
    """
    def __init__(self, input_dim=768, hidden_dim=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )

        # Tree nodes
        self.node_low_high = nn.Linear(hidden_dim, 2)  # G1+G2 vs G3+G4
        self.node_g1_g2 = nn.Linear(hidden_dim, 2)      # G1 vs G2
        self.node_g3_g4 = nn.Linear(hidden_dim, 2)      # G3 vs G4

    def forward(self, x):
        h = self.shared(x)

        # 每层输出都可独立解释
        low_high = self.node_low_high(h)     # 第一步:哪个大类?
        g1_g2 = self.node_g1_g2(h)            # 如果是低级别: G1 or G2?
        g3_g4 = self.node_g3_g4(h)            # 如果是高级别: G3 or G4?

        # 最终4分类logits
        logits = torch.stack([
            g1_g2[:, 0],   # G1
            g1_g2[:, 1],   # G2
            g3_g4[:, 0],   # G3
            g3_g4[:, 1],   # G4
        ], dim=1)

        return logits, {"low_high": low_high, "g1_g2": g1_g2, "g3_g4": g3_g4}

# ===================== 训练与评估 =====================

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        if isinstance(model, TreeStructuredClassifier):
            logits, _ = model(x)
        else:
            logits = model(x)

        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / len(loader), correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels, all_probs = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if isinstance(model, TreeStructuredClassifier):
            logits, _ = model(x)
        else:
            logits = model(x)

        loss = criterion(logits, y)
        total_loss += loss.item()
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y.cpu().tolist())
        all_probs.extend(torch.softmax(logits, 1).cpu().tolist())
    return total_loss / len(loader), correct / total, all_preds, all_labels, all_probs

def run_experiment(model_name, model_cls, X, y, device, n_splits=5):
    """5折交叉验证"""
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 处理类别不平衡: 对G4上采样
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64)

        model = model_cls().to(device)
        # 类别权重处理G4稀少
        class_weights = torch.FloatTensor([1.0, 1.0, 2.0, 5.0]).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

        best_acc = 0
        for epoch in range(50):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc, *_ = evaluate(model, val_loader, criterion, device)
            if val_acc > best_acc:
                best_acc = val_acc

        _, _, preds, labels, probs = evaluate(model, val_loader, criterion, device)
        results.append({"fold": fold, "preds": preds, "labels": labels, "probs": probs, "best_acc": best_acc})

    return results

# ===================== 主程序 =====================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 加载数据
    X = np.load(FEATURE_DIR / "text_features.npy")
    y = np.load(FEATURE_DIR / "labels.npy")
    print(f"Features: {X.shape}, Labels: {dict(zip(*np.unique(y, return_counts=True)))}")

    grade_names = ["G1", "G2", "G3", "G4"]

    print("\n=== MLP Baseline ===")
    mlp_results = run_experiment("MLP", MLPClassifier, X, y, device)

    print("\n=== Tree-Structured Classifier ===")
    tree_results = run_experiment("Tree", TreeStructuredClassifier, X, y, device)

    # 汇总结果
    for name, res in [("MLP", mlp_results), ("Tree", tree_results)]:
        all_preds = sum([r["preds"] for r in res], [])
        all_labels = sum([r["labels"] for r in res], [])
        all_probs = sum([r["probs"] for r in res], [])

        acc = sum(1 for p, l in zip(all_preds, all_labels) if p == l) / len(all_labels)
        print(f"\n=== {name} ===")
        print(f"Accuracy: {acc:.3f}")
        print(classification_report(all_labels, all_preds, target_names=grade_names, zero_division=0))

        # Per-class AUC (OvR)
        try:
            y_bin = label_binarize(all_labels, classes=[0,1,2,3])
            aucs = []
            for i in range(4):
                if len(np.unique(y_bin[:, i])) > 1:
                    aucs.append(roc_auc_score(y_bin[:, i], [p[i] for p in all_probs]))
            print(f"AUC: {np.mean(aucs):.3f} (per-class: {[f'{a:.3f}' for a in aucs]})")
        except Exception as e:
            print(f"AUC: N/A ({e})")

    print("\n=== DONE ===")
