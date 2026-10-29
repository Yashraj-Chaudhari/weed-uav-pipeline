#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from weed_uav_pipeline.inference import WeedPredictor
from weed_uav_pipeline.planner import treatment_waypoints


def main():
    parser = argparse.ArgumentParser(description="Predict weed mask and export a Gazebo path CSV")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask-out")
    parser.add_argument("--path-out")
    parser.add_argument("--field-width-m", type=float, required=True)
    parser.add_argument("--field-height-m", type=float, required=True)
    parser.add_argument("--grid-rows", type=int, default=3)
    parser.add_argument("--grid-cols", type=int, default=3)
    parser.add_argument("--min-weed-fraction", type=float, default=0.05)
    parser.add_argument("--altitude-m", type=float, default=2.5)
    parser.add_argument("--origin-x", type=float, default=0.0)
    parser.add_argument("--origin-y", type=float, default=0.0)
    args = parser.parse_args()

    image = Path(args.image)
    mask_path = Path(args.mask_out) if args.mask_out else Path("output") / (image.stem + "_weed_mask.png")
    path_path = Path(args.path_out) if args.path_out else Path("output") / (image.stem + "_path.csv")
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    path_path.parent.mkdir(parents=True, exist_ok=True)
    mask = WeedPredictor(args.checkpoint).predict_file(str(image), str(mask_path))
    points = treatment_waypoints(
        mask, args.field_width_m, args.field_height_m,
        args.grid_rows, args.grid_cols, args.min_weed_fraction,
        args.altitude_m, args.origin_x, args.origin_y, args.origin_x, args.origin_y,
    )

    cell_w = args.field_width_m / args.grid_cols
    cell_h = args.field_height_m / args.grid_rows
    with path_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["visit", "grid_row", "grid_col", "x_m", "y_m", "z_m", "weed_fraction"])
        for visit, (x, y, z, fraction) in enumerate(points, 1):
            col = min(args.grid_cols - 1, max(0, int((x - args.origin_x) / cell_w)))
            row = min(args.grid_rows - 1, max(0, args.grid_rows - 1 - int((y - args.origin_y) / cell_h)))
            writer.writerow([visit, row, col, "%.4f" % x, "%.4f" % y, "%.3f" % z, "%.5f" % fraction])

    print("Mask:", mask_path)
    print("Path:", path_path)
    print("Treatment points:", len(points))


if __name__ == "__main__":
    main()

