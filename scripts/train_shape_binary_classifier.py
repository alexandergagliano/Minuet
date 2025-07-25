import torch
import torch.nn as nn
import numpy as np
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve,
    roc_auc_score, average_precision_score
)

def build_targets(feat):
    if feat == "merging_either_fraction":
        major = val_set["from_file_b"]["merging_major-disturbance_fraction"][:].astype(np.float32)
        minor = val_set["from_file_b"]["merging_minor-disturbance_fraction"][:].astype(np.float32)
        none  = val_set["from_file_b"]["merging_none_fraction"][:].astype(np.float32)
        labels = np.full(major.shape, np.nan, dtype=np.float32)
        labels[(major > 0.2) | (minor > 0.2)] = 1.0
        labels[none > 0.5] = 0.
        # All others (ambiguous) remain np.nan
        return labels

    elif feat.endswith("_yes_fraction"):
        base = feat.replace("_yes_fraction", "")
        yes = val_set["from_file_b"][f"{base}_yes_fraction"][:].astype(np.float32)
        no  = val_set["from_file_b"][f"{base}_no_fraction"][:].astype(np.float32)

        labels = np.full(yes.shape, np.nan, dtype=np.float32)

        # Assign positive (1) where "yes" > 0.5, negative (0) where "no" > 0.5
        labels[yes > 0.5] = 1.0
        labels[no > 0.5] = 0.0

        # All others (ambiguous) remain np.nan
        return labels

    else:
        return val_set["from_file_b"][feat][:].astype(np.float32)

# ---- Plot Style ----
def plot_config():
    sns.set_context("poster")
    sns.set(font_scale=2)
    sns.set_style("white")
    sns.set_style("ticks", {"xtick.major.size": 15, "ytick.major.size": 20})
    sns.set_style("ticks", {"xtick.minor.size": 8, "ytick.minor.size": 8})
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Palatino"],
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 15,
        "ytick.major.size": 20,
        "xtick.minor.size": 8,
        "ytick.minor.size": 8,
    })
plot_config()

latent_dim = 5
print("Loading data…")
val_set = h5py.File(
    "/Users/alexgagliano/Documents/Research/GalaxyAutoencoder/"
    "data/test_subset/maskeddaep_results_gzsample_1752682352_maskdaep5_crossmatch.h5", "r"
)
latents = val_set["from_file_a"]["latent"][:].squeeze()

PRETTY = {
    "merging_either_fraction":           "Merger",  
    "how-rounded_round_fraction":        "Round",
    "has-spiral-arms_yes_fraction":      "Has Spiral Arms",
    "disk-edge-on_yes_fraction":         "Edge‑on Disk",
}

shape_feats = [
    "merging_either_fraction",           
    "has-spiral-arms_yes_fraction",
    "how-rounded_round_fraction",
    "disk-edge-on_yes_fraction",
]

# ---- Model ----
class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        h = self.norm(x)
        h = self.act(self.lin1(h))
        h = self.drop(h)
        h = self.lin2(h)
        return x + h

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden=512, blocks=4, dropout=0.1):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU())
        self.blocks = nn.ModuleList([ResBlock(hidden, dropout) for _ in range(blocks)])
        self.out = nn.Linear(hidden, 1)
    def forward(self, x):
        x = self.inp(x)
        for blk in self.blocks:
            x = blk(x)
        return self.out(x).squeeze(1)

def train_one_feature(lat_np, tgt_np, epochs=200, lr=1e-3,
                      cap_pw=500., batch=256, sampler=True, verbose=False):
    x_tr, x_te, y_tr, y_te = train_test_split(
        lat_np, tgt_np, test_size=0.2, stratify=np.round(tgt_np), random_state=42
    )
    x_tr, x_te = map(torch.tensor, (x_tr, x_te))
    y_tr, y_te = map(torch.tensor, (y_tr, y_te))
    x_tr = x_tr.float(); x_te = x_te.float()
    y_tr = y_tr.float(); y_te = y_te.float()
    mean, std = x_tr.mean(0, keepdim=True), x_tr.std(0, keepdim=True).clamp_min(1e-6)
    x_tr = (x_tr - mean)/std; x_te = (x_te - mean)/std
    pw = (len(y_tr) - y_tr.sum()).item() / (y_tr.sum().item() + 1e-6)
    pw = min(pw, cap_pw)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw]))
    if sampler:
        weights = torch.where(y_tr > 0.5, torch.tensor(pw), torch.tensor(1.0))
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_tr, y_tr),
            batch_size=batch,
            sampler=torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=True)
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_tr, y_tr),
            batch_size=batch, shuffle=True
        )
    model = MLPClassifier(latent_dim)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train()
        tot = 0.
        for xb, yb in train_loader:
            loss = criterion(model(xb), yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            opt.step()
            tot += loss.item()*len(xb)
        if verbose and ep % 10 == 0:
            print(f"  Epoch {ep:02d} | Loss {tot/len(y_tr):.4f}")
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(x_te)).cpu().numpy()
    return probs, y_te.cpu().numpy()

def mean_std(curves, xs):
    vals = [np.interp(xs, a, b) for a, b in curves]
    vals = np.vstack(vals)
    return vals.mean(0), vals.std(0)

skf = StratifiedKFold(5, shuffle=True, random_state=42)

all_lbls = []
all_global_preds = []
all_global_trues = []

cv_cm = {}
roc_curves = {}
pr_curves = {}
roc_auc_avgs = {}
pr_auc_avgs = {}

for feat in shape_feats:
    print(f"\n=== {PRETTY[feat]} ===")
    t_full = build_targets(feat)

    #subset where labels are confident 
    mask = ~np.isnan(t_full)
    latents_used = latents[mask]
    t_full_used = t_full[mask]
    print("Class Balance...Mergers:", np.sum(t_full_used == 1), "Non-mergers:", np.sum(t_full_used == 0))
    y_round = np.round(t_full_used)

    fold_preds = []
    fold_trues = []
    fold_rocs = []
    fold_prs = []
    cm_folds = []
    # cross-val
    for tr, te in skf.split(latents_used, y_round):
        x = latents_used[tr]
        y = t_full_used[tr]
        p, y_true = train_one_feature(x, y, epochs=500, verbose=True)
        y_bin = np.round(y_true)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            continue
        # Save for global/global AUC and PRC
        fold_preds.append(p)
        fold_trues.append(y_bin)
        # ROC/PR curves per fold (for mean±σ)
        fpr, tpr, _ = roc_curve(y_bin, p)
        fold_rocs.append((fpr, tpr))
        prec, rec, _ = precision_recall_curve(y_bin, p)
        order = np.argsort(rec)
        fold_prs.append((rec[order], prec[order]))
        cm_folds.append(confusion_matrix(y_bin, (p > 0.5), normalize="true"))
    # Store per-feature results
    roc_curves[feat] = fold_rocs
    pr_curves[feat]  = fold_prs
    cv_cm[feat]      = cm_folds
    all_lbls.append(PRETTY[feat])
    # flatten all folds for global ROC/PR
    all_global_preds.append(np.concatenate(fold_preds))
    all_global_trues.append(np.concatenate(fold_trues))

    # Plot confusion matrix (mean±std across folds)
    if cm_folds:
        cms = np.stack(cm_folds)
        cm_mean, cm_std = cms.mean(0), cms.std(0)
        fig, ax = plt.subplots()
        sns.heatmap(cm_mean, vmin=0, vmax=1, cmap="Blues",
                    xticklabels=["False", "True"], yticklabels=["False", "True"],
                    cbar=False, annot=False, ax=ax)
        for (i, j), v in np.ndenumerate(cm_mean):
            ax.text(j+0.5, i+0.5, f"{v:.2f}±{cm_std[i,j]:.2f}", ha="center", va="center",
                    color="white" if v > .5 else "black")
        ax.grid(False)
        plt.title(PRETTY[feat])
        plt.tight_layout(); plt.savefig(f"cv_confmat_{feat}.png"); plt.close()

# ------------- CV ROC (mean±σ, AUC in legend) -------------
plt.figure(figsize=(10, 8))
for feat in shape_feats:
    curves = roc_curves[feat]
    if not curves: continue
    xs = np.linspace(0, 1, 200)
    mu, sd = mean_std(curves, xs)
    # mean AUC across folds for legend
    aucs = [np.trapz(b, a) for a, b in curves]
    mean_auc = np.mean(aucs)
    plt.plot(xs, mu, label=f"{PRETTY[feat]} (AUC={mean_auc:.2f})")
    plt.fill_between(xs, mu-sd, mu+sd, alpha=0.2)
plt.plot([0,1], [0,1], 'k--', label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig("cv_roc_mean_std.png")
plt.close()

# ------------- CV PRC (mean±σ, AUC in legend) -------------
plt.figure(figsize=(10, 8))
for feat in shape_feats:
    curves = pr_curves[feat]
    if not curves: continue
    xs = np.linspace(0, 1, 200)
    mu, sd = mean_std(curves, xs)
    # mean AUC (AUC-PRC) across folds for legend
    aps = [np.trapz(b, a) for a, b in curves]
    mean_ap = np.mean(aps)
    plt.plot(xs, mu, label=f"{PRETTY[feat]} (AUC={mean_ap:.2f})")
    plt.fill_between(xs, mu - sd, mu + sd, alpha=0.2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.tight_layout()
plt.savefig("cv_pr_mean_std.png")
plt.close()

# ------------- Global ROC (all folds concatenated) -------------
plt.figure(figsize=(10, 8))
for preds, trues, lbl in zip(all_global_preds, all_global_trues, all_lbls):
    fpr, tpr, _ = roc_curve(trues, preds)
    auc = roc_auc_score(trues, preds)
    plt.plot(fpr, tpr, label=f"{lbl} (AUC={auc:.2f})")
plt.plot([0,1], [0,1], 'k--', label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig("global_roc_curve.png")
plt.close()

# ------------- Global PR (all folds concatenated) -------------
plt.figure(figsize=(10, 8))
for preds, trues, lbl in zip(all_global_preds, all_global_trues, all_lbls):
    prec, rec, _ = precision_recall_curve(trues, preds)
    ap = average_precision_score(trues, preds)
    plt.plot(rec, prec, label=f"{lbl} (AUC={ap:.2f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.tight_layout()
plt.savefig("global_pr_curve.png")
plt.close()

