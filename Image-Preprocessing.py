import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch

# --- Hardware Detection ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device detected: {device.upper()}")


def preprocess_fracture_data(src_root, dest_folder, target_size=(224, 224)):
    # 1. Setup Directories
    src_root = Path(src_root)
    dest_folder = Path(dest_folder)
    dest_folder.mkdir(parents=True, exist_ok=True)

    # Supported formats
    valid_extensions = ('.jpg', '.jpeg', '.png')
    
    # 2. Find all images
    image_paths = [p for p in src_root.rglob('*') if p.suffix.lower() in valid_extensions]
    print(f"Found {len(image_paths)} images. Starting processing...")

    # CLAHE Setup
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Stats Trackers (for Mean/Std)
    n_pixels = 0
    sum_pixels = 0.0
    sum_sq_pixels = 0.0

    # 3. Processing Loop
    for i, img_path in enumerate(tqdm(image_paths)):
        try:
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # A. Grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # B. CLAHE (Contrast Enhancement)
            enhanced = clahe.apply(gray)

            # C. Noise Control (Median Blur - Edge Preserving)
            denoised = cv2.medianBlur(enhanced, 3)

            # D. Resize
            resized = cv2.resize(denoised, target_size, interpolation=cv2.INTER_AREA)

            # E. Save with unique suffix to avoid duplicates
            new_name = f"bone_{i:05d}_{img_path.name}"
            save_path = dest_folder / new_name
            cv2.imwrite(str(save_path), resized)

            # F. Update Stats (Normalized to [0, 1] range)
            img_normalized = resized.astype(np.float32) / 255.0
            sum_pixels += np.sum(img_normalized)
            sum_sq_pixels += np.sum(np.square(img_normalized))
            n_pixels += img_normalized.size

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    # 4. Compute Dataset Mean and Std
    final_mean = sum_pixels / n_pixels
    final_std = np.sqrt((sum_sq_pixels / n_pixels) - (final_mean ** 2))

    print("\n--- Processing Complete ---")
    print(f"Images saved to: {dest_folder}")
    print(f"Dataset Mean: {final_mean:.4f}")
    print(f"Dataset Std:  {final_std:.4f}")
  
    return final_mean, final_std

# --- Execution ---
# Replace these paths with your actual folder locations
SOURCE_DIR = r'C:\Users\preeti\Desktop\muramskxrays\MURA-v1.1\MURA-v1.1\train'
OUTPUT_DIR = r'C:\Users\preeti\Desktop\clahe'

mean, std = preprocess_fracture_data(SOURCE_DIR, OUTPUT_DIR)
