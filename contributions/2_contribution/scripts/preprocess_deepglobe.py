import os
import random
import shutil
import re
import argparse

import numpy as np
from PIL import Image
from tqdm import tqdm


COLORMAP = {
    (255, 255, 255): 5,     # Red → Class 0
    (0, 255, 0): 3,     # Green → Class 1
    (0, 0, 255): 4,     # Blue → Class 2
    (255, 255, 0): 1,   # Yellow → Class 3
    (255, 0, 255): 2,   # Magenta → Class 4
    (0, 255, 255): 0,   # Cyan → Class 5
    (0, 0, 0): 6        # Black → Unknown (class to ignore)
}


def organize_images_by_class(folder):
    for fname in os.listdir(folder):
        if fname.endswith(".tif"):
            # Extract class name: everything before the first digit
            match = re.match(r"([a-zA-Z]+)", fname)
            if not match:
                print(f"Skipping: {fname} (no class name found)")
                continue

            class_name = match.group(1)
            class_dir = os.path.join(folder, class_name)
            os.makedirs(class_dir, exist_ok=True)

            src_path = os.path.join(folder, fname)
            dst_path = os.path.join(class_dir, fname)

            shutil.move(src_path, dst_path)
            print(f"Moved: {fname} → {class_dir}")


def convert_colored_mask_to_class_ids(mask_rgb, colormap):
    """
    Converts an RGB mask (H, W, 3) to class ID mask (H, W).
    """
    h, w, _ = mask_rgb.shape
    mask_class = np.zeros((h, w), dtype=np.uint8)

    for color, class_id in colormap.items():
        r, g, b = color
        match = (mask_rgb[:, :, 0] == r) & (mask_rgb[:, :, 1] == g) & (mask_rgb[:, :, 2] == b)
        mask_class[match] = class_id

    return mask_class


def extract_tiles_and_labels(folder, tile_size=256, num_classes=6, threshold=10, ignore_class=6):
    """
    folder: path to directory containing *_sat.jpg and *_mask.png files
    tile_size: size of square tiles to extract
    num_classes: number of valid semantic classes (excluding the ignored one)
    threshold: minimum pixel count per class to be considered present
    ignore_class: class index to ignore in label computation
    """
    image_files = [f for f in os.listdir(folder) if f.endswith("_sat.jpg")]

    tile_images = []
    tile_labels = []

    for img_file in tqdm(image_files, desc="Processing images"):
        base_name = img_file.replace("_sat.jpg", "")
        mask_file = f"{base_name}_mask.png"
        img_path = os.path.join(folder, img_file)
        mask_path = os.path.join(folder, mask_file)

        if not os.path.exists(mask_path):
            print(f"Warning: Mask not found for {img_file}")
            continue

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        image_np = np.array(image)
        mask_np = convert_colored_mask_to_class_ids(np.array(mask), COLORMAP)

        h, w = mask_np.shape
        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                if y + tile_size > h or x + tile_size > w:
                    continue

                img_tile = image_np[y:y + tile_size, x:x + tile_size, :]
                mask_tile = mask_np[y:y + tile_size, x:x + tile_size]

                label = np.zeros(num_classes, dtype=np.uint8)
                for cls in range(num_classes):  # only valid classes
                    if np.sum(mask_tile == cls) >= threshold:
                        label[cls] = 1

                # Skip if tile has only "unknown" (class 6)
                if np.sum(label) == 0 and np.all(mask_tile == ignore_class):
                    continue

                tile_images.append(img_tile)
                tile_labels.append(label)

    return tile_images, tile_labels


def save_tiles_with_labels(image_tiles, label_vectors, out_dir, prefix="image"):
    """
    Save tiles with multi-hot label vector encoded in the filename.
    
    image_tiles: list of (H, W, 3) numpy arrays
    label_vectors: list of binary vectors (e.g. [1, 0, 1, 0, 0, 1])
    out_dir: target directory to save the images
    prefix: optional prefix for image filenames
    """
    for i, (tile, label) in enumerate(zip(image_tiles, label_vectors)):
        label_str = ''.join(map(str, label.tolist()))
        filename = f"{prefix}{i:06d}_{label_str}.jpg"
        filepath = os.path.join(out_dir, filename)

        img = Image.fromarray(tile)
        img.save(filepath, format="JPEG", quality=95)


def filter_balanced_multilabel_subset(images, labels, seed=42):
    """
    Selects a balanced subset of image-label pairs:
    - 10k randomly from single-label samples
    - 10k randomly from two-label samples
    - All from three or more labels
    """
    assert len(images) == len(labels), "Image-label list mismatch."

    label_sums = np.array([label.sum() for label in labels])

    # Get indices
    idx_1 = np.where(label_sums == 1)[0]
    idx_2 = np.where(label_sums == 2)[0]
    idx_multi = np.where(label_sums > 2)[0]

    # Sample 10k randomly (without replacement if possible)
    random.seed(seed)
    idx_1_sampled = random.sample(list(idx_1), min(10000, len(idx_1)))
    idx_2_sampled = random.sample(list(idx_2), min(10000, len(idx_2)))

    # Combine all selected indices
    selected_indices = idx_1_sampled + idx_2_sampled + list(idx_multi)

    # Extract filtered lists
    filtered_images = [images[i] for i in selected_indices]
    filtered_labels = [labels[i] for i in selected_indices]

    print(f"Selected {len(filtered_images)} samples (1-label: {len(idx_1_sampled)}, 2-label: {len(idx_2_sampled)}, >2-label: {len(idx_multi)})")

    return filtered_images, filtered_labels


def find_train_dir(root: str) -> str:
    # Prefer explicit train folders if present
    for cand in ("train", "Train", "training", "Training"):
        cpath = os.path.join(root, cand)
        if os.path.isdir(cpath):
            return cpath
    # Otherwise, search for a folder containing *_sat.jpg and *_mask.png
    for dirpath, _, filenames in os.walk(root):
        has_sat = any(f.endswith("_sat.jpg") for f in filenames)
        has_mask = any(f.endswith("_mask.png") for f in filenames)
        if has_sat and has_mask:
            return dirpath
    raise FileNotFoundError("Could not find DeepGlobe train folder with _sat.jpg/_mask.png files.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess DeepGlobe into multi-label tiles")
    parser.add_argument("--source_path", type=str, default=None, help="Path to raw DeepGlobe folder")
    parser.add_argument("--destination_path", type=str, required=True, help="Output root (will create images/)")
    parser.add_argument("--tile_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_kagglehub", action="store_true", help="Download via kagglehub")
    parser.add_argument(
        "--kaggle_dataset",
        type=str,
        default="balraj98/deepglobe-land-cover-classification-dataset",
    )
    args = parser.parse_args()

    source_root = args.source_path
    if args.use_kagglehub:
        try:
            import kagglehub
        except Exception as exc:
            raise ImportError(
                "kagglehub is required. Install with `pip install kagglehub[pandas-datasets]`."
            ) from exc
        source_root = kagglehub.dataset_download(args.kaggle_dataset)

    if source_root is None:
        raise ValueError("Provide --source_path or use --use_kagglehub.")

    train_dir = find_train_dir(source_root)
    out_dir = os.path.join(args.destination_path, "images")
    os.makedirs(out_dir, exist_ok=True)

    images, labels = extract_tiles_and_labels(train_dir, tile_size=args.tile_size)
    images_subset, labels_subset = filter_balanced_multilabel_subset(images, labels, seed=args.seed)
    save_tiles_with_labels(images_subset, labels_subset, out_dir)

    print(f"Saved DeepGlobe tiles to {out_dir}")


if __name__ == "__main__":
    main()
