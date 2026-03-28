import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from research_finetune import FractureNet


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward(self, x):
        logits, fused_map = self.model(x)

        # Hook gradients
        fused_map.register_hook(self.save_gradient)

        self.activations = fused_map
        return logits

    def generate(self, class_idx):
        # 1. Get gradients
        grads = self.gradients            # [1, C, H, W]
        acts = self.activations           # [1, C, H, W]

        # 2. Global average pooling on gradients
        weights = torch.mean(grads, dim=(2, 3))  # [1, C]

        # 3. Weighted combination
        cam = torch.zeros(acts.shape[2:], dtype=torch.float32).to(acts.device)

        for i in range(weights.shape[1]):
            cam += weights[0, i] * acts[0, i]

        # 4. ReLU
        cam = torch.relu(cam)

        # 5. Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.detach().cpu().numpy()

    def get_bounding_box(self, cam, threshold=0.6):
        """
        Convert heatmap to bounding box
        """
        cam_uint8 = np.uint8(255 * cam)

        # Threshold
        _, thresh = cv2.threshold(cam_uint8, int(255 * threshold), 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return None

        # Take largest contour (most confident region)
        largest = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(largest)

        return x, y, w, h


def predict_and_gradcam(image_path, model_path, mean, std):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = FractureNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    gradcam = GradCAM(model)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([mean], [std])
    ])

    raw_img = Image.open(image_path).convert('L')
    input_tensor = transform(raw_img).unsqueeze(0).to(device)

    # Forward pass (no torch.no_grad!)
    logits = gradcam.forward(input_tensor)

    probs = F.softmax(logits, dim=1)
    conf, pred = torch.max(probs, dim=1)

    # Backward pass
    model.zero_grad()
    logits[0, pred].backward()

    # Generate CAM
    cam = gradcam.generate(pred.item())

    # Resize CAM to original image size
    cam = cv2.resize(cam, (224, 224))

    # Get bounding box
    bbox = gradcam.get_bounding_box(cam)

    img_resized = raw_img.resize((224, 224))
    img_np = np.array(img_resized)
    img_color = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(img_color, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Convert CAM to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(
        cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB), 0.6,
        heatmap, 0.4, 0
    )

    classes = ['Normal', 'Fracture']
    result_text = f"{classes[pred.item()]} ({conf.item() * 100:.2f}%)"
    print(result_text)

    # Plot results
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(img_np, cmap='gray')
    plt.title("Original")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap)
    plt.title("Grad-CAM Heatmap")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title("Overlay with Heatmap")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# --- RUN ---
MEAN, STD = 0.45, 0.22
MODEL_PATH = "best_fracture_model.pth"
IMAGE_PATH = "path/to/test_image.jpg"

predict_and_gradcam(IMAGE_PATH, MODEL_PATH, MEAN, STD)