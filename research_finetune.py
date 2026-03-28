import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, cohen_kappa_score, classification_report, recall_score
from sklearn.model_selection import train_test_split

# --- 1. MODEL ---


class FractureNet(nn.Module):
    def __init__(self, pretrain_path="fused_encoder_weights.pth"):
        super().__init__()

        from preprocess_dataset import ConvMAE
        base = ConvMAE()

        try:
            ckpt = torch.load(pretrain_path, map_location='cpu')
            base.stage1.load_state_dict(ckpt['stage1'])
            base.stage2.load_state_dict(ckpt['stage2'])
            base.stage3_proj.load_state_dict(ckpt['stage3_proj'])
            base.transformer.load_state_dict(ckpt['transformer'])
            base.fusion.load_state_dict(ckpt['fusion'])
            print("✅ Pretrained weights loaded")
        except Exception as e:
            print("⚠️ Training from scratch:", e)

        self.encoder = nn.ModuleDict({
            's1': base.stage1,
            's2': base.stage2,
            's3p': base.stage3_proj,
            'tr': base.transformer,
            'fs': base.fusion,
            'p1': base.proj1,
            'p2': base.proj2
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
        f1_raw = self.encoder['s1'](x)
        f2_raw = self.encoder['s2'](f1_raw)
        s3_feat = self.encoder['s3p'](f2_raw)

        b, c, h, w = s3_feat.shape
        tokens = s3_feat.flatten(2).transpose(1, 2)
        s3_out = self.encoder['tr'](tokens).transpose(1, 2).view(b, c, h, w)

        f1 = self.encoder['p1'](f1_raw)
        f2 = self.encoder['p2'](f2_raw)
        fused_map = self.encoder['fs'](f1, f2, s3_out)

        logits = self.classifier(fused_map)

        return logits, fused_map   # keep for Grad-CAM


# --- 2. TRAINER ---
class FractureTrainer:
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, 2.5]).to(device)
        )

    def get_optimizer(self, lr, mode='heads_only'):
        if mode == 'heads_only':
            return optim.AdamW(self.model.classifier.parameters(), lr=lr)
        else:
            return optim.AdamW([
                {'params': self.model.encoder.parameters(), 'lr': lr * 0.1},
                {'params': self.model.classifier.parameters(), 'lr': lr}
            ])

    def train_epoch(self, optimizer, scaler, use_amp):
        self.model.train()
        total_loss = 0

        for imgs, labels in tqdm(self.train_loader, leave=False):
            imgs, labels = imgs.to(self.device), labels.to(self.device)

            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                logits, _ = self.model(imgs)
                loss = self.criterion(logits, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        all_preds, all_labels = [], []
        total_loss = 0

        for imgs, labels in self.val_loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)

            logits, _ = self.model(imgs)
            loss = self.criterion(logits, labels)

            preds = torch.argmax(logits, dim=1)

            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        return (
            np.array(all_preds),
            np.array(all_labels),
            total_loss / len(self.val_loader)
        )


# --- 3. MAIN ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler(enabled=use_amp)

    DATA_PATH = "processed_data"
    MEAN, STD = 0.45, 0.22

    # ✅ Separate transforms
    train_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((224, 224)),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([MEAN], [STD])
    ])

    val_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([MEAN], [STD])
    ])

    full_dataset = datasets.ImageFolder(DATA_PATH)

    # ✅ Stratified split
    targets = full_dataset.targets
    train_idx, val_idx = train_test_split(
        range(len(targets)),
        test_size=0.2,
        stratify=targets,
        random_state=42
    )

    train_set = Subset(
        datasets.ImageFolder(DATA_PATH, transform=train_transform),
        train_idx
    )

    val_set = Subset(
        datasets.ImageFolder(DATA_PATH, transform=val_transform),
        val_idx
    )

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

    model = FractureNet().to(device)
    trainer = FractureTrainer(model, train_loader, val_loader, device)

    # --- STAGE 1 ---
    print("\n🔥 STAGE 1: HEAD WARMUP")
    for p in model.encoder.parameters():
        p.requires_grad = False

    optimizer = trainer.get_optimizer(1e-3, 'heads_only')

    for epoch in range(5):
        loss = trainer.train_epoch(optimizer, scaler, use_amp)
        print(f"Warmup Epoch {epoch + 1} | Loss: {loss:.4f}")

    # --- STAGE 2 ---
    print("\n🔥 STAGE 2: FULL FINETUNE")
    for p in model.encoder.parameters():
        p.requires_grad = True

    optimizer = trainer.get_optimizer(1e-4, 'full')
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)

    best_f1 = 0
    patience = 0

    for epoch in range(50):
        train_loss = trainer.train_epoch(optimizer, scaler, use_amp)
        preds, labels, val_loss = trainer.evaluate()

        f1 = f1_score(labels, preds)
        recall = recall_score(labels, preds)
        kappa = cohen_kappa_score(labels, preds)

        print(f"\nEpoch {epoch + 1}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"F1: {f1:.4f} | Recall: {recall:.4f} | Kappa: {kappa:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "best_fracture_model.pth")
            patience = 0
            print("⭐ Best model saved")
        else:
            patience += 1

        if patience >= 10:
            print("⛔ Early stopping")
            break

        scheduler.step()

    # --- FINAL REPORT ---
    preds, labels, _ = trainer.evaluate()
    print("\n📊 FINAL REPORT")
    print(classification_report(labels, preds, target_names=['Normal', 'Fracture']))


if __name__ == "__main__":
    main()
