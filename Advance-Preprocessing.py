import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch

# --- Hardware Detection ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device detected: {device.upper()}")


def is_bad_image(img):
    """Detect low-quality or unusable images"""
    if img is None or img.size == 0:
        return True, "Unreadable"

    mean = img.mean()
    std = img.std()

    if mean < 20:
        return True, "Too Dark"
    elif mean > 235:
        return True, "Too Bright"
    elif std < 10:
        return True, "Low Contrast"

    return False, "Good"


def preprocess_fracture_data_clean(
    src_root,
    dest_folder,
    target_size=(224, 224),
    save_bad=False
):
    src_root = Path(src_root)
    dest_folder = Path(dest_folder)

    clean_dir = dest_folder / "clean"
    bad_dir = dest_folder / "bad"

    clean_dir.mkdir(parents=True, exist_ok=True)
    if save_bad:
        bad_dir.mkdir(parents=True, exist_ok=True)

    valid_extensions = ('.jpg', '.jpeg', '.png')

    image_paths = [p for p in src_root.rglob('*') if p.suffix.lower() in valid_extensions]

    print(f"Found {len(image_paths)} images. Starting processing...")

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Stats
    n_pixels = 0
    sum_pixels = 0.0
    sum_sq_pixels = 0.0

    # Counters
    kept = 0
    removed = 0

    # Log file
    log_file = dest_folder / "bad_images_log.txt"

    with open(log_file, "w") as log:

        for i, img_path in enumerate(tqdm(image_paths)):
            try:
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

                # --- Quality Check ---
                bad, reason = is_bad_image(img)

                if bad:
                    removed += 1
                    log.write(f"{img_path} --> {reason}\n")

                    if save_bad and img is not None:
                        cv2.imwrite(str(bad_dir / f"bad_{i}.png"), img)

                    continue

                # --- Processing ---
                enhanced = clahe.apply(img)
                denoised = cv2.medianBlur(enhanced, 3)
                resized = cv2.resize(denoised, target_size, interpolation=cv2.INTER_AREA)

                # Save clean image
                save_path = clean_dir / f"bone_{kept:05d}.png"
                cv2.imwrite(str(save_path), resized)

                kept += 1

                # --- Stats ---
                img_norm = resized.astype(np.float32) / 255.0
                sum_pixels += np.sum(img_norm)
                sum_sq_pixels += np.sum(img_norm ** 2)
                n_pixels += img_norm.size

            except Exception as e:
                removed += 1
                log.write(f"{img_path} --> ERROR: {e}\n")

    # --- Final Stats ---
    mean = sum_pixels / n_pixels
    std = np.sqrt((sum_sq_pixels / n_pixels) - (mean ** 2))

    print("\n--- Processing Complete ---")
    print(f"✅ Clean images: {kept}")
    print(f"❌ Removed images: {removed}")
    print(f"📁 Saved to: {clean_dir}")
    print(f"📄 Log file: {log_file}")
    print(f"\nDataset Mean: {mean:.4f}")
    print(f"Dataset Std:  {std:.4f}")

    return mean, std

SOURCE_DIR = r'C:\Users\preeti\Desktop\muramskxrays\MURA-v1.1\MURA-v1.1\train'
OUTPUT_DIR = r'C:\Users\preeti\Desktop\clahe_clean'

mean, std = preprocess_fracture_data_clean(
    SOURCE_DIR,
    OUTPUT_DIR,
    save_bad=True   # saves bad images for inspection
)
