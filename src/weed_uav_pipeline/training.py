"""Training pipeline for the WeedMap RedEdge RGB tiles and color annotations."""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .model import WeedUNet, vegetation_features


def find_pairs(root, max_black_fraction=0.40):
    root = Path(root)
    pairs = []
    for label in sorted(root.glob("*/groundtruth/*_GroundTruth_color.png")):
        flight = label.parent.parent
        prefix = flight.name + "_"
        if not label.name.startswith(prefix):
            continue
        frame = label.name[len(prefix):-len("_GroundTruth_color.png")]
        image = flight / "tile" / "RGB" / (frame + ".png")
        binary_gt = label.with_name(label.name[:-len("_GroundTruth_color.png")] + ".png")
        if not image.is_file() or not binary_gt.is_file():
            continue
        raw = cv2.imread(str(binary_gt), cv2.IMREAD_GRAYSCALE)
        if raw is None or float(np.mean(raw == 0)) >= max_black_fraction:
            continue
        pairs.append((str(image), str(label), flight.name))
    if not pairs:
        raise RuntimeError("No paired WeedMap RGB/groundtruth tiles found; check --data-root")
    return pairs


class WeedMapDataset(Dataset):
    def __init__(self, pairs, augment=False):
        self.pairs = pairs
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    @staticmethod
    def read_pair(pair):
        image_path, label_path, _ = pair
        bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        label_bgr = cv2.imread(label_path, cv2.IMREAD_COLOR)
        if bgr is None or label_bgr is None:
            raise RuntimeError("Could not read paired image or label: " + image_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        label = cv2.cvtColor(label_bgr, cv2.COLOR_BGR2RGB)
        if label.shape[:2] != rgb.shape[:2]:
            label = cv2.resize(label, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        # The notebook maps the WeedMap red annotation to its weed class.
        red_weed = (label[:, :, 0] > 200) & (label[:, :, 1] < 80) & (label[:, :, 2] < 80)
        target = np.where(red_weed, 0, 1).astype(np.int64)
        return rgb, target

    def __getitem__(self, index):
        rgb, target = self.read_pair(self.pairs[index])
        if self.augment:
            if random.random() < 0.5:
                rgb, target = np.fliplr(rgb), np.fliplr(target)
            if random.random() < 0.5:
                rgb, target = np.flipud(rgb), np.flipud(target)
        return torch.from_numpy(np.ascontiguousarray(rgb)), torch.from_numpy(np.ascontiguousarray(target))


def split_pairs(pairs, seed=42, validation_fraction=0.2):
    """Hold out complete flights when possible to limit adjacent-frame leakage."""
    groups = sorted(set(pair[2] for pair in pairs))
    rng = random.Random(seed)
    if len(groups) > 1:
        rng.shuffle(groups)
        count = max(1, min(len(groups) - 1, round(len(groups) * validation_fraction)))
        validation_groups = set(groups[:count])
        train = [p for p in pairs if p[2] not in validation_groups]
        val = [p for p in pairs if p[2] in validation_groups]
    else:
        ordered = sorted(pairs, key=lambda p: p[0])
        count = max(1, min(len(ordered) - 1, round(len(ordered) * validation_fraction)))
        train, val = ordered[:-count], ordered[-count:]
    if not train or not val:
        raise RuntimeError("Need at least two paired tiles for a train/validation split")
    return train, val


def evaluate(model, loader, device):
    model.eval()
    confusion = torch.zeros((2, 2), dtype=torch.int64)
    with torch.no_grad():
        for rgb, target in loader:
            features = vegetation_features(rgb).to(device)
            target = target.to(device)
            prediction = model(features).argmax(dim=1)
            bins = torch.bincount((target * 2 + prediction).reshape(-1), minlength=4)
            confusion += bins.reshape(2, 2).cpu()
    tp = confusion[0, 0].item()
    fp = confusion[1, 0].item()
    fn = confusion[0, 1].item()
    total = confusion.sum().item()
    iou = tp / max(tp + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    accuracy = (confusion[0, 0] + confusion[1, 1]).item() / max(total, 1)
    return {"accuracy": accuracy, "weed_iou": iou, "weed_precision": precision, "weed_recall": recall}


def main():
    parser = argparse.ArgumentParser(description="Train the project weed segmentation model")
    parser.add_argument("--data-root", required=True, help="FYP/Tiles/RedEdge")
    parser.add_argument("--output", default="weed_uav_pipeline/checkpoints/weed_unet.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.0007)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pairs = find_pairs(args.data_root)
    train_pairs, val_pairs = split_pairs(pairs, args.seed)
    train_set, val_set = WeedMapDataset(train_pairs, True), WeedMapDataset(val_pairs)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WeedUNet().to(device)

    # Mild inverse-frequency weights, computed only from training labels.
    counts = np.zeros(2, dtype=np.int64)
    for pair in train_pairs:
        _, target = WeedMapDataset.read_pair(pair)
        counts += np.bincount(target.ravel(), minlength=2)
    weights = np.sqrt(counts.sum() / (2.0 * np.maximum(counts, 1)))
    weights = np.clip(weights, 0.5, 5.0)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_iou, output = -1.0, Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    print("Tiles:", len(pairs), "train:", len(train_pairs), "validation:", len(val_pairs), "device:", device)
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, batches = 0.0, 0
        for rgb, target in train_loader:
            features = vegetation_features(rgb).to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batches += 1
        metrics = evaluate(model, val_loader, device)
        print("epoch %d/%d loss %.4f val acc %.4f weed_iou %.4f precision %.4f recall %.4f" %
              (epoch, args.epochs, running_loss / max(batches, 1),
               metrics["accuracy"], metrics["weed_iou"],
               metrics["weed_precision"], metrics["weed_recall"]))
        if metrics["weed_iou"] > best_iou:
            best_iou = metrics["weed_iou"]
            torch.save({"state_dict": model.state_dict(), "classes": ["weed", "other"],
                        "input": "ExG,ExGR,CIVE", "best_validation": metrics}, output)
    print("Saved best validation checkpoint:", output)


if __name__ == "__main__":
    main()

