import json
import os
from PIL import Image

DATA_DIR = "dataset"
SPLITS = ["train", "val", "test"]
MANIFESTS = {split: f"{DATA_DIR}/{split}.jsonl" for split in SPLITS}

def find_leftmost_xy(img, target_rgb):
    """Return the smallest x such that img pixel == target_rgb."""
    pixels = img.load()
    width, height = img.size
    min_x = width
    for x in range(width):
        for y in range(height):
            if pixels[x, y] == target_rgb:
                min_x = x
                return min_x
    return None

# Define RGB tuples for detection
RED = (255, 0, 0)
BLUE = (0, 0, 255)

errors = []
for split, manifest_path in MANIFESTS.items():
    with open(manifest_path) as f:
        for line_num, line in enumerate(f, 1):
            entry = json.loads(line)
            img_path = os.path.join(DATA_DIR, entry["image_path"])
            label = entry["completion"].strip()  # "yes" or "no"
            img = Image.open(img_path)
            red_x = find_leftmost_xy(img, RED)
            blue_x = find_leftmost_xy(img, BLUE)
            if red_x is None or blue_x is None:
                errors.append((entry["image_path"], "missing shape"))
            else:
                is_left = red_x < blue_x
                if is_left != (label == "yes"):
                    errors.append((entry["image_path"], f"label says {label}, but red_x={red_x}, blue_x={blue_x}"))

if not errors:
    print("✅ All labels match image positions!")
else:
    print("❌ Found mismatches:")
    for img, msg in errors:
        print(f"  {img}: {msg}")
