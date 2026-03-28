import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # ✅ NEW
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# --- 1. Weighted Fusion Module ---


class WeightedFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # ❌ OLD: scalar weights
        # self.weights = nn.Parameter(torch.ones(3) / 3)

        # ✅ NEW: channel-wise weights (better)
        self.weights = nn.Parameter(torch.ones(3, channels, 1, 1))

    def forward(self, f1, f2, f3):
        w = torch.softmax(self.weights, dim=0)
        return w[0] * f1 + w[1] * f2 + w[2] * f3


# --- 2. Encoder Components ---
class ConvStage(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),

            # ❌ OLD: BatchNorm
            # nn.BatchNorm2d(out_ch),

            # ✅ NEW: GroupNorm (better for medical/small batch)
            nn.GroupNorm(8, out_ch),

            nn.GELU()
        )

    def forward(self, x):
        return self.conv(x)


class ConvMAE(nn.Module):
    def __init__(self, img_size=224, patch_size=16, mask_ratio=0.75):
        super().__init__()
        self.mask_ratio = mask_ratio

        # Stage 1 & 2
        self.stage1 = ConvStage(1, 64, stride=2)   # 224 -> 112
        self.stage2 = ConvStage(64, 128, stride=2)  # 112 -> 56

        # Stage 3
        self.stage3_proj = nn.Conv2d(128, 256, kernel_size=4, stride=4)  # 56->14

        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=4)

        # Add positional embedding here
        self.pos_embed = nn.Parameter(torch.zeros(1, 196, 256))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Fusion
        self.fusion = WeightedFusion(256)
        self.proj1 = nn.Sequential(nn.AdaptiveAvgPool2d((14, 14)), nn.Conv2d(64, 256, 1))
        self.proj2 = nn.Sequential(nn.AdaptiveAvgPool2d((14, 14)), nn.Conv2d(128, 256, 1))

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=4),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    # ❌ OLD: pixel masking
    # def random_masking(self, x):

    # ✅ NEW: token masking (MAE CORRECT)
    def random_masking(self, tokens):
        B, N, C = tokens.shape
        num_masked = int(self.mask_ratio * N)

        mask = torch.zeros(B, N, device=tokens.device)

        for i in range(B):
            perm = torch.randperm(N, device=tokens.device)
            mask[i, perm[:num_masked]] = 1

        # apply mask on tokens
        tokens = tokens * (1 - mask.unsqueeze(-1))

        return tokens, mask

    def forward(self, x):
        # 1. Encoding
        s1 = self.stage1(x)
        s2 = self.stage2(s1)

        # 2. Patchify
        s3_feat = self.stage3_proj(s2)  # (B, 256, 14, 14)

        b, c, h, w = s3_feat.shape

        # ❌ OLD: masking on feature map
        # masked_s3, mask = self.random_masking(s3_feat)

        tokens = s3_feat.flatten(2).transpose(1, 2)  # (B, 196, 256)

        # ✅ Step 1: masking
        tokens, mask = self.random_masking(tokens)

        # ✅ Step 2: add positional embedding
        tokens = tokens + self.pos_embed

        # 3. Pass through Transformer
        s3_out = self.transformer(tokens).transpose(1, 2).view(b, c, h, w)

        # 4. Fusion
        f1 = self.proj1(s1)
        f2 = self.proj2(s2)
        fused = self.fusion(f1, f2, s3_out)

        # 5. Reconstruction
        recon = self.decoder(fused)

        # ✅ NEW: mask resize for loss
        mask_2d = mask.view(b, 1, h, w)
        mask_up = F.interpolate(mask_2d, size=(224, 224), mode='nearest')

        return recon, mask_up


# --- 3. Training Loop ---
def train_convmae(data_dir, mean, std):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    epochs = 50  # ephocs should be 400
    batch_size = 64
    lr = 1.5e-4
    weight_decay = 0.05

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([mean], [std])
    ])

    # ❌ OLD: empty dataset
    # dataset = torch.utils.data.Dataset

    # ✅ FIX: use ImageFolder
    from torchvision.datasets import ImageFolder
    dataset = ImageFolder(data_dir, transform=transform)

    # ✅ FIX: DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=4, pin_memory=True, persistent_workers=True)

    model = ConvMAE().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler(device='cuda')
    model.train()
    for epoch in range(epochs):
        loop = tqdm(loader, leave=True)

        for imgs, _ in loop:
            imgs = imgs.to(device, non_blocking=True)

            with torch.amp.autocast(device_type='cuda'):
                recon, mask = model(imgs)

                # ✅ FIX: correct masked loss
                loss = criterion(recon * mask, imgs * mask) / model.mask_ratio

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loop.set_description(f"Epoch [{epoch}/{epochs}]")
            loop.set_postfix(loss=loss.item())

    torch.save(model.state_dict(), "convmae_pretrain_full.pth")

    # ✅ FIX: include stage3_proj
    torch.save({
        'stage1': model.stage1.state_dict(),
        'stage2': model.stage2.state_dict(),
        'stage3_proj': model.stage3_proj.state_dict(),
        'transformer': model.transformer.state_dict(),
        'fusion': model.fusion.state_dict()
    }, "fused_encoder_weights.pth")

if __name__ == "__main__":
    MY_DATA_PATH = r'C:\Users\preeti\Desktop\clahe'
    MY_MEAN = 0.2475
    MY_STD = 0.1855
    # --- STEP 3: START PRE-TRAINING ---
    train_convmae(
        data_dir=MY_DATA_PATH,
        mean=MY_MEAN,
        std=MY_STD
    )
