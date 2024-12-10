import json
import random
import shutil
from pathlib import Path

# Paths
metadata_file = "simple_scatter_data1/scatter_metadata.json"
image_folder = Path("simple_scatter_data1")  # Folder containing all images
output_dir = Path("../data/scatterdata/scatter")
images_dir = output_dir / "images"
annotations_dir = output_dir / "annotations"

# Create output directories
images_dir.mkdir(parents=True, exist_ok=True)
annotations_dir.mkdir(parents=True, exist_ok=True)

train_folder = images_dir / "train"
val_folder = images_dir / "val"
train_folder.mkdir(exist_ok=True)
val_folder.mkdir(exist_ok=True)

TRAIN_RATIO = 0.8
train_file = annotations_dir / "train.json"
val_file = annotations_dir / "val.json"

# Load metadata
with open(metadata_file, "r") as f:
    metadata = json.load(f)

# Extract images and annotations
images = metadata["images"]
annotations = metadata["annotations"]

# Shuffle and split
random.seed(42)
random.shuffle(images)

# Calculate split indices
num_train = int(len(images) * TRAIN_RATIO)
train_images = images[:num_train]
train_images.sort(key=lambda img: img["id"])
val_images = images[num_train:]
val_images.sort(key=lambda img: img["id"])

# Create mappings for fast annotation lookup
image_id_to_annotations = {img["id"]: [] for img in images}
for ann in annotations:
    image_id_to_annotations[ann["image_id"]].append(ann)

# Split annotations based on image splits
train_annotations = []
val_annotations = []

for img in train_images:
    train_annotations.extend(image_id_to_annotations[img["id"]])

for img in val_images:
    val_annotations.extend(image_id_to_annotations[img["id"]])

# Create train and validation metadata
train_metadata = {
    "images": train_images,
    "annotations": train_annotations,
    "categories": metadata["categories"]
}

val_metadata = {
    "images": val_images,
    "annotations": val_annotations,
    "categories": metadata["categories"]
}

# Save train and validation JSON
with open(train_file, "w") as f:
    json.dump(train_metadata, f, indent=4)

with open(val_file, "w") as f:
    json.dump(val_metadata, f, indent=4)

print(f"JSON splits created:")
print(f"- Train JSON: {train_file} ({len(train_images)} images, {len(train_annotations)} annotations)")
print(f"- Validation JSON: {val_file} ({len(val_images)} images, {len(val_annotations)} annotations)")

# Copy images to respective folders
for img in train_images:
    src = image_folder / img["file_name"]
    dst = train_folder / img["file_name"]
    shutil.copy(src, dst)

for img in val_images:
    src = image_folder / img["file_name"]
    dst = val_folder / img["file_name"]
    shutil.copy(src, dst)

print(f"Images split into folders:")
print(f"- Train folder: {train_folder} ({len(train_images)} images)")
print(f"- Validation folder: {val_folder} ({len(val_images)} images)")
