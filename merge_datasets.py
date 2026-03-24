import os
import shutil
from pathlib import Path

# 🔥 DATASET PATHS
datasets = [
    r"C:/Users/dell/Downloads/tyre.v1i.yolov8",
    r"C:/Users/dell/Downloads/wheel.v1i.yolov8",
    r"C:/Users/dell/Downloads/Vehicle Wheel Detection.v1i.yolov8"
]

output = r"C:/Users/dell/Downloads/dataset_final"

splits = ["train", "valid", "test"]

# Create output folders
for split in splits:
    for folder in ["images", "labels"]:
        Path(output, split, folder).mkdir(parents=True, exist_ok=True)

count = 0

for dataset in datasets:
    dataset_name = Path(dataset).name  # ⭐ ONLY folder name

    for split in splits:
        img_dir = Path(dataset) / split / "images"
        lbl_dir = Path(dataset) / split / "labels"

        if not img_dir.exists():
            continue

        for img_file in img_dir.glob("*.*"):
            new_name = f"{dataset_name}_{count}{img_file.suffix}"

            dst_img = Path(output) / split / "images" / new_name
            shutil.copy(img_file, dst_img)

            lbl_file = lbl_dir / (img_file.stem + ".txt")
            if lbl_file.exists():
                dst_lbl = Path(output) / split / "labels" / f"{dataset_name}_{count}.txt"
                shutil.copy(lbl_file, dst_lbl)

            count += 1

print("✅ Datasets merged successfully!")