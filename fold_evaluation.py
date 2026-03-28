import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
# from tqdm import tqdm
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, confusion_matrix, cohen_kappa_score,
    roc_auc_score, recall_score
)

# ---------------- MODEL ----------------


class FractureNet(nn.Module):
    def __init__(self, pretrain_path="fused_encoder_weights.pth"):
        super().__init__()
        from preprocess_dataset import ConvMAE
        base = ConvMAE()

        try:
            ckpt = torch.load(pretrain_path, map_location='cpu')
            base.load_state_dict(ckpt, strict=False)
            print("✅ Pretrained weights loaded")
        except (FileNotFoundError, RuntimeError):
            print("⚠️ Training from scratch")

        self.encoder = nn.ModuleDict({
            's1': base.stage1, 's2': base.stage2,
            's3p': base.stage3_proj, 'tr': base.transformer,
            'fs': base.fusion, 'p1': base.proj1, 'p2': base.proj2
        })

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        f1 = self.encoder['s1'](x)
        f2 = self.encoder['s2'](f1)
        s3 = self.encoder['s3p'](f2)

        b, c, h, w = s3.shape
        tokens = s3.flatten(2).transpose(1, 2)
        s3 = self.encoder['tr'](tokens).transpose(1, 2).view(b, c, h, w)

        f1p = self.encoder['p1'](f1)
        f2p = self.encoder['p2'](f2)
        fused = self.encoder['fs'](f1p, f2p, s3)

        logits = self.classifier(fused)
        return logits, fused


# ---------------- TRAINER ----------------
class Trainer:
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, 2.5]).to(device)
        )

        self.use_amp = torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)

    def train_epoch(self, optimizer):
        self.model.train()
        total_loss = 0

        for imgs, labels in self.train_loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)

            with torch.amp.autocast(device_type='cuda', enabled=self.use_amp):
                logits, _ = self.model(imgs)
                loss = self.criterion(logits, labels)

            optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        all_preds, all_labels, all_probs = [], [], []

        for imgs, labels in self.val_loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)

            logits, _ = self.model(imgs)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

        return np.array(all_preds), np.array(all_labels), np.array(all_probs)


# ---------------- MAIN EXPERIMENT ----------------
def run_experiment(data_dir, mean, std):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ Separate transforms
    train_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((224, 224)),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([mean], [std])
    ])

    val_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([mean], [std])
    ])

    base_dataset = datasets.ImageFolder(data_dir)
    labels = [label for _, label in base_dataset.samples]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n🔥 FOLD {fold + 1}")

        train_set = Subset(
            datasets.ImageFolder(data_dir, transform=train_transform),
            train_idx
        )
        val_set = Subset(
            datasets.ImageFolder(data_dir, transform=val_transform),
            val_idx
        )

        train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

        model = FractureNet().to(device)
        trainer = Trainer(model, train_loader, val_loader, device)

        # -------- STAGE 1 (FREEZE) --------
        for p in model.encoder.parameters():
            p.requires_grad = False

        optimizer = optim.AdamW(model.classifier.parameters(), lr=1e-3)

        for _ in range(3):
            trainer.train_epoch(optimizer)

        # -------- STAGE 2 (UNFREEZE) --------
        for p in model.encoder.parameters():
            p.requires_grad = True

        optimizer = optim.AdamW([
            {'params': model.encoder.parameters(), 'lr': 1e-5},
            {'params': model.classifier.parameters(), 'lr': 1e-4}
        ])

        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)

        best_f1 = 0
        patience = 5
        counter = 0

        for epoch in range(30):
            trainer.train_epoch(optimizer)
            preds, targets, probs = trainer.evaluate()

            f1 = f1_score(targets, preds)

            if f1 > best_f1:
                best_f1 = f1
                best_preds, best_targets, best_probs = preds, targets, probs
                counter = 0
            else:
                counter += 1

            if counter >= patience:
                print("⛔ Early stopping")
                break

            scheduler.step()

        # -------- METRICS --------
        cm = confusion_matrix(best_targets, best_preds)
        tn, fp, fn, tp = cm.ravel()

        sens = recall_score(best_targets, best_preds)
        spec = tn / (tn + fp + 1e-8)

        fold_metrics.append({
            'f1': f1_score(best_targets, best_preds),
            'sens': sens,
            'spec': spec,
            'kappa': cohen_kappa_score(best_targets, best_preds),
            'auc': roc_auc_score(best_targets, best_probs)
        })

        print(f"Fold {fold + 1} F1: {fold_metrics[-1]['f1']:.4f}")

    # -------- FINAL SUMMARY --------
    print("\n📊 FINAL RESULTS (5-FOLD)")
    for key in fold_metrics[0]:
        values = [f[key] for f in fold_metrics]
        print(f"{key.upper():<6}: {np.mean(values):.4f} ± {np.std(values):.4f}")


# ---------------- RUN ----------------
if __name__ == "__main__":
    run_experiment("path/to/dataset", 0.45, 0.22)
