import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import os
import matplotlib.pyplot as plt

# --- 1. Weighted Fusion ---


class WeightedFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(3, channels, 1, 1))

    def forward(self, f1, f2, f3):
        w = torch.softmax(self.weights, dim=0)
        return w[0] * f1 + w[1] * f2 + w[2] * f3

# --- 2. Encoder Blocks ---


class ConvStage(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.GELU()
        )

    def forward(self, x):
        return self.conv(x)

# --- 3. ConvMAE Model ---


class ConvMAE(nn.Module):
    def __init__(self, mask_ratio=0.75):
        super().__init__()
        self.mask_ratio = mask_ratio

        # Encoder
        self.stage1 = ConvStage(1, 64, 2)
        self.stage2 = ConvStage(64, 128, 2)
        self.stage3_proj = nn.Conv2d(128, 256, kernel_size=4, stride=4)

        # Transformer
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=4)

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, 196, 256))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Mask token for MAE
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 256))

        # Fusion
        self.fusion = WeightedFusion(256)
        self.proj1 = nn.Sequential(nn.AdaptiveAvgPool2d((14, 14)), nn.Conv2d(64, 256, 1))
        self.proj2 = nn.Sequential(nn.AdaptiveAvgPool2d((14, 14)), nn.Conv2d(128, 256, 1))

        # Lighter decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 4),
            nn.GELU(),
            nn.Conv2d(128, 1, 3, padding=1),
            nn.Sigmoid()
        )

    # --- Vectorized token masking ---
    def random_masking(self, tokens):
        B, N, C = tokens.shape
        num_masked = int(self.mask_ratio * N)

        noise = torch.rand(B, N, device=tokens.device)
        ids_shuffle = torch.argsort(noise, dim=1)

        mask = torch.zeros(B, N, device=tokens.device)
        mask.scatter_(1, ids_shuffle[:, :num_masked], 1)

        tokens = tokens * (1 - mask.unsqueeze(-1)) + self.mask_token * mask.unsqueeze(-1)
        return tokens, mask

    def forward(self, x):
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3_proj(s2)

        B, C, H, W = s3.shape
        tokens = s3.flatten(2).transpose(1, 2)
        tokens, mask = self.random_masking(tokens)
        tokens = tokens + self.pos_embed
        tokens = self.transformer(tokens)
        s3_out = tokens.transpose(1, 2).view(B, C, H, W)

        fused = self.fusion(self.proj1(s1), self.proj2(s2), s3_out)
        recon = self.decoder(fused)
        mask_up = F.interpolate(mask.view(B, 1, H, W), size=(224, 224), mode='nearest')

        return recon, mask_up

# --- 4. Dataset ---


class FlatImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("L")
        except Exception:
            img = Image.new("L", (224, 224))
        if self.transform:
            img = self.transform(img)
        return img, 0

# --- 5. Visualization Helper ---


def save_preview(model, imgs, epoch, device, mean, std):
    model.eval()
    with torch.no_grad():
        test_img = imgs[0:1].to(device)
        recon, mask = model(test_img)
        recon = F.interpolate(recon, size=(224, 224), mode='bilinear', align_corners=False)
        orig = (test_img[0].cpu().squeeze() * std + mean).clamp(0, 1)
        rec = (recon[0].cpu().squeeze() * std + mean).clamp(0, 1)
        masked_view = orig.clone()
        masked_view[mask[0].cpu().squeeze() > 0.5] = 0

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(orig, cmap='gray'); axes[0].set_title("Original")
        axes[1].imshow(masked_view, cmap='gray'); axes[1].set_title("Masked Input")
        axes[2].imshow(rec, cmap='gray'); axes[2].set_title("Reconstructed")
        for ax in axes:
            ax.axis('off')
        plt.savefig(f"preview_epoch_{epoch}.png")
        plt.close()
    model.train()

# --- 6. Training ---


def train_convmae(data_dir, mean, std):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # Hyperparameters
    epochs = 50
    batch_size = 16
    lr = 1.5e-4
    save_every = 5

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([mean], [std])
    ])

    dataset = FlatImageDataset(data_dir, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=4, pin_memory=True)     

    model = ConvMAE().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    start_epoch = 0
    if os.path.exists("last_checkpoint.pth"):
        checkpoint = torch.load("last_checkpoint.pth")
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        model.train()
        loop = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")
        total_loss = 0

        for imgs, _ in loop:
            imgs = imgs.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                recon, mask = model(imgs)

            recon = F.interpolate(recon, size=(224, 224), mode='bilinear', align_corners=False)

            loss = ((recon - imgs) ** 2 * mask).sum() / mask.sum()

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

        scheduler.step()
        save_preview(model, imgs, epoch, device, mean, std)

        if epoch % save_every == 0 or epoch == epochs - 1:
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
            }, "last_checkpoint.pth")
            torch.save(model.state_dict(), f"convmae_epoch_{epoch}.pth")

    print("Pre-training Complete!")

# --- RUN ---
if __name__ == "__main__":
    train_convmae(
        data_dir=r'C:\Users\preeti\Desktop\clahe',
        mean=0.2808,
        std=0.2252
    )
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
