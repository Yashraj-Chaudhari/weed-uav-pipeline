"""Project mask (0=weed, nonzero=crop/soil) to a grid of ROS field targets."""
from math import hypot


def cell_weed_fractions(mask, rows, cols):
    if mask is None or len(mask) == 0 or len(mask[0]) == 0 or any(len(row) != len(mask[0]) for row in mask):
        raise ValueError("mask must be a non-empty rectangular 2D array")
    if rows <= 0 or cols <= 0:
        raise ValueError("grid dimensions must be positive")
    height, width = len(mask), len(mask[0])
    result = []
    for r in range(rows):
        y0, y1 = r * height // rows, (r + 1) * height // rows
        row_result = []
        for c in range(cols):
            x0, x1 = c * width // cols, (c + 1) * width // cols
            pixels = [mask[y][x] for y in range(y0, y1) for x in range(x0, x1)]
            row_result.append(sum(int(v) == 0 for v in pixels) / float(len(pixels)) if pixels else 0.0)
        result.append(row_result)
    return result


def treatment_waypoints(mask, field_width_m, field_height_m, grid_rows=3, grid_cols=3,
                        min_weed_fraction=0.05, altitude_m=2.5, origin_x=0.0,
                        origin_y=0.0, start_x=0.0, start_y=0.0):
    """Return nearest-neighbor ordered (x,y,z,weed_fraction) for qualifying cells.

    Assumes an orthographic top-down image. Image row zero maps to the far (+y)
    edge; ROS x/y increase right/up from the configured field origin.
    """
    if field_width_m <= 0 or field_height_m <= 0:
        raise ValueError("field dimensions must be positive")
    if not 0 <= min_weed_fraction <= 1:
        raise ValueError("weed threshold must be between 0 and 1")
    fractions = cell_weed_fractions(mask, grid_rows, grid_cols)
    cw, ch = field_width_m / grid_cols, field_height_m / grid_rows
    pending = []
    for r in range(grid_rows):
        for c in range(grid_cols):
            fraction = fractions[r][c]
            if fraction >= min_weed_fraction:
                pending.append((origin_x + (c + .5) * cw,
                                origin_y + (grid_rows - r - .5) * ch,
                                altitude_m, fraction))
    ordered, x, y = [], start_x, start_y
    while pending:
        i = min(range(len(pending)), key=lambda n: hypot(pending[n][0]-x, pending[n][1]-y))
        point = pending.pop(i)
        ordered.append(point)
        x, y = point[0], point[1]
    return ordered
