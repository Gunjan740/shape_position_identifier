import os
import json
import random
from PIL import Image, ImageDraw

# ——— Configuration ———
OUTPUT_DIR = "dataset"
SPLITS = {"train": 100, "val": 20, "test": 20}
CANVAS_SIZE = (64, 64)
MARGIN = 10
SHAPE_SIZE = 20  # diameter of circle and side length of square
MID_X = CANVAS_SIZE[0] // 2
COLORS = {"circle": "#FF0000", "square": "#0000FF"}
PROMPT = "Is the circle left to the square? →"

random.seed(42)

# ——— Helpers ———
def make_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for split in SPLITS:
        os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

def sample_positions(label):
    """Return ((cx, cy), (sx, sy)) for circle & square centers."""
    # x‐ranges
    left_x_range = (MARGIN, MID_X - MARGIN)
    right_x_range = (MID_X + MARGIN, CANVAS_SIZE[0] - MARGIN)
    # pick which shape goes left/right
    if label == "yes":
        cx = random.randint(*left_x_range)
        sx = random.randint(*right_x_range)
    else:
        sx = random.randint(*left_x_range)
        cx = random.randint(*right_x_range)
    # y‐positions free in [MARGIN, height-MARGIN]
    y_low, y_high = MARGIN, CANVAS_SIZE[1] - MARGIN
    cy = random.randint(y_low, y_high)
    sy = random.randint(y_low, y_high)
    return (cx, cy), (sx, sy)

def draw_and_save(image_idx, split, label):
    # create blank canvas
    img = Image.new("RGB", CANVAS_SIZE, color="white")
    draw = ImageDraw.Draw(img)

    # get centers
    (cx, cy), (sx, sy) = sample_positions(label)

    # draw circle
    r = SHAPE_SIZE // 2
    draw.ellipse(
        [(cx - r, cy - r), (cx + r, cy + r)],
        fill=COLORS["circle"]
    )

    # draw square
    half = SHAPE_SIZE // 2
    draw.rectangle(
        [(sx - half, sy - half), (sx + half, sy + half)],
        fill=COLORS["square"]
    )

    # save file
    fname = f"img_{image_idx:05d}.png"
    path = os.path.join(OUTPUT_DIR, split, fname)
    img.save(path)
    return os.path.join(split, fname)  # relative path for manifest

# ——— Main Generation Loop ———
def generate_dataset():
    make_dirs()
    for split, count in SPLITS.items():
        manifest = []
        for i in range(1, count + 1):
            label = random.choice(["yes", "no"])
            img_path = draw_and_save(i, split, label)
            manifest.append({
                "image_path": img_path,
                "prompt": PROMPT,
                "completion": f" {label}"
            })
        # write JSONL
        with open(os.path.join(OUTPUT_DIR, f"{split}.jsonl"), "w") as f:
            for entry in manifest:
                f.write(json.dumps(entry) + "\n")

if __name__ == "__main__":
    generate_dataset()
    print("Done: images and manifests generated under ./dataset/")
