import json
import os

# 1) Update these to match your repo details:
GITHUB_USER = "Gunjan740"
REPO_NAME   = "shape_position_identifier"
BRANCH      = "main"

# Base raw URL to your dataset folder:
GITHUB_RAW = f"https://raw.githubusercontent.com/{GITHUB_USER}/{REPO_NAME}/{BRANCH}/dataset"

# 2) Paths:
DATASET_DIR = "dataset"
SPLITS      = ("train", "val", "test")

for split in SPLITS:
    in_path  = os.path.join(DATASET_DIR, f"{split}.jsonl")
    out_path = os.path.join(DATASET_DIR, f"{split}_ft.jsonl")

    with open(in_path, "r") as fin, open(out_path, "w") as fout:
        for line in fin:
            entry = json.loads(line)
            local_path = entry["image_path"]            # e.g. "train/img_00007.png"
            url        = f"{GITHUB_RAW}/{local_path}"   # full raw.githubusercontent URL

            messages = [
                {"role": "system",    "content": "You are a shape classifier."},
                {"role": "user",      "content": [
                    {"type": "image_url", "image_url": {"url": url}}
                ]},
                {"role": "assistant", "content": entry["completion"]}
            ]

            fout.write(json.dumps({"messages": messages}) + "\n")

    print(f"Wrote → {out_path}")
