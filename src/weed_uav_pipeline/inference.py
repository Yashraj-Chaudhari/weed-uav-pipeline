"""Load a trained checkpoint and predict a mono8 mask (0=weed, 255=other)."""
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .model import WeedUNet, vegetation_features


class WeedPredictor:
    def __init__(self, checkpoint, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        saved = torch.load(checkpoint, map_location=self.device)
        self.model = WeedUNet().to(self.device)
        self.model.load_state_dict(saved["state_dict"])
        self.model.eval()

    def predict_rgb(self, rgb):
        if rgb is None or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Expected an RGB image with three channels")
        h, w = rgb.shape[:2]
        pad_h, pad_w = (-h) % 16, (-w) % 16
        tensor = torch.from_numpy(np.ascontiguousarray(rgb)).unsqueeze(0)
        features = vegetation_features(tensor).to(self.device)
        if pad_h or pad_w:
            features = F.pad(features, (0, pad_w, 0, pad_h), mode="replicate")
        with torch.no_grad():
            labels = self.model(features).argmax(dim=1)[0, :h, :w].cpu().numpy()
        return np.where(labels == 0, 0, 255).astype(np.uint8)

    def predict_file(self, image_path, mask_path):
        bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError("Could not read image: " + image_path)
        mask = self.predict_rgb(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if not cv2.imwrite(mask_path, mask):
            raise OSError("Could not write mask: " + mask_path)
        return mask

