"""
DNDF (Deep Neural Decision Forest) vs MLP 对比实验
实现简化的可微分决策树进行OSCC分级
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

class SimpleDNDF(nn.Module):
    """简化版深度神经决策森林：1棵树, depth=3, 8个叶子节点对应G1-G4
    每个分裂节点使用sigmoid可微路由, 叶子存储类别概率分布"""
    def __init__(self, in_dim=1024, depth=3, num_classes=4, hidden=128):
        super().__init__()
        self.depth = depth
        self.num_leaves = 2 ** depth  # 8 leaves
        self.num_classes = num_classes

        # Shared feature extractor (降维)
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden * 2),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden * 2, hidden), nn.ReLU()
        )

        # Split nodes: for a complete binary tree of depth 3, there are 2^depth - 1 = 7 split nodes
        self.num_splits = 2**depth - 1  # 7
        self.split_weights = nn.Parameter(torch.randn(self.num_splits, hidden) * 0.01)
        self.split_bias = nn.Parameter(torch.zeros(self.num_splits))

        # Leaf nodes: probability distribution over classes
        self.leaf_probs = nn.Parameter(torch.randn(self.num_leaves, num_classes) * 0.01)

    def forward(self, x):
        h = self.shared(x)  # (B, hidden)
        B = h.size(0)

        # Compute routing probabilities for each leaf
        leaf_probs = torch.zeros(B, self.num_leaves, device=x.device)

        for leaf_idx in range(self.num_leaves):
            prob = torch.ones(B, device=x.device)
            current_idx = 0  # Start from root

            for d in range(self.depth):
                # Determine if this leaf goes left or right at depth d
                go_right = (leaf_idx >> (self.depth - 1 - d)) & 1
                node_logit = torch.mm(h, self.split_weights[current_idx].unsqueeze(1)).squeeze(1) + self.split_bias[current_idx]
                split_prob = torch.sigmoid(node_logit)

                if go_right:
                    prob = prob * split_prob
                    current_idx = 2 * current_idx + 2  # right child
                else:
                    prob = prob * (1 - split_prob)
                    current_idx = 2 * current_idx + 1  # left child

            leaf_probs[:, leaf_idx] = prob

        # Normalize routing probabilities
        leaf_probs = leaf_probs / leaf_probs.sum(dim=1, keepdim=True)

        # Weighted sum of leaf distributions
        logits = torch.mm(leaf_probs, torch.softmax(self.leaf_probs, dim=-1))
        return logits, leaf_probs  # return routing for interpretability

class MLPBaseline(nn.Module):
    def __init__(self, in_dim=1024, hidden=256, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.BatchNorm1d(hidden), nn.Dropout(0.4),
            nn.Linear(hidden, hidden//2), nn.ReLU(), nn.BatchNorm1d(hidden//2), nn.Dropout(0.4),
            nn.Linear(hidden//2, num_classes)
        )

    def forward(self, x):
        return self.net(x), None

def train_epoch(model, loader, opt, crit, device):
    model.train()
    L_t, C_t, T_t = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits, _ = model(x)
        loss = crit(logits, y)
        loss.backward()
        opt.step()
        L_t += loss.item(); C_t += (logits.argmax(1)==y).sum().item(); T_t += y.size(0)
    return L_t/len(loader), C_t/T_t

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    P, L, R = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        P.extend(logits.argmax(1).cpu().tolist())
        L.extend(y.cpu().tolist())
        R.extend(torch.softmax(logits,1).cpu().tolist())
    return P, L, R

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load data
    X = np.load(DATA_DIR / "X_img_uni.npy")
    y_all = np.load(DATA_DIR / "y.npy")

    print(f"Features: {X.shape}")
    print(f"Label distribution: {dict(zip(*np.unique(y_all, return_counts=True)))}")

    kf = StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED)
    results = {}

    for name, model_cls in [("MLP", MLPBaseline), ("DNDF", SimpleDNDF)]:
        P, L, R = [], [], []
        for tr, val in kf.split(X, y_all):
            Xt, Xv = X[tr], X[val]; yt, yv = y_all[tr], y_all[val]

            # Class weights for imbalance
            counts = np.bincount(yt, minlength=4)
            weights = torch.FloatTensor([max(counts)/c if c>0 else 1.0 for c in counts]).to(device)

            tr_ds = TensorDataset(torch.FloatTensor(Xt), torch.LongTensor(yt))
            v_ds = TensorDataset(torch.FloatTensor(Xv), torch.LongTensor(yv))
            tr_ldr = DataLoader(tr_ds, 32, True); v_ldr = DataLoader(v_ds, 64)

            model = model_cls().to(device)
            crit = nn.CrossEntropyLoss(weight=weights)
            opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

            for ep in range(100):
                train_epoch(model, tr_ldr, opt, crit, device)

            p, l, r = evaluate(model, v_ldr, device)
            P.extend(p); L.extend(l); R.extend(r)

        acc = sum(1 for a,b in zip(P,L) if a==b) / len(L)
        print(f"\n=== {name} (4-class G1-G4) ===")
        print(f"Accuracy: {acc:.3f}")
        print(classification_report(L, P, target_names=["G1","G2","G3","G4"], zero_division=0))
        try:
            from sklearn.preprocessing import label_binarize
            yb = label_binarize(L, classes=[0,1,2,3])
            aucs = [roc_auc_score(yb[:,i], [r[i] for r in R]) for i in range(4) if len(np.unique(yb[:,i]))>1]
            print(f"AUC: {np.mean(aucs):.3f} ({[f'{a:.3f}' for a in aucs]})")
        except: pass
        results[name] = acc

    print(f"\n=== Summary ===")
    for n, a in results.items():
        print(f"  {n}: Acc={a:.3f}")
    print(f"\n  DNDF has interpretable routing paths through {SimpleDNDF(1024).num_leaves} leaf nodes.")
    print(f"  Each leaf stores a learned grade probability distribution.")
