import os
import shutil
import hashlib
from pathlib import Path
from PIL import Image, UnidentifiedImageError

# =========================
# CONFIGURATION
# =========================

# Folder containing merged class folders
SOURCE_DATASET = Path("../merged_dataset")

# Folder where cleaned images will be copied
CLEANED_DATASET = Path("../cleaned_dataset")

# Folder where suspicious/small images will be moved for manual review
REVIEW_DATASET = Path("../review_dataset")

# Allowed image extensions
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Minimum acceptable width/height
MIN_WIDTH = 100
MIN_HEIGHT = 100

# If True, exact duplicate images are skipped
REMOVE_DUPLICATES = True

# If True, non-image files are ignored/skipped
SKIP_NON_IMAGES = True

# =========================
# HELPER FUNCTIONS
# =========================

def create_folders():
    """Create output folders."""
    CLEANED_DATASET.mkdir(parents=True, exist_ok=True)
    REVIEW_DATASET.mkdir(parents=True, exist_ok=True)

    for class_dir in SOURCE_DATASET.iterdir():
        if class_dir.is_dir():
            (CLEANED_DATASET / class_dir.name).mkdir(parents=True, exist_ok=True)
            (REVIEW_DATASET / class_dir.name).mkdir(parents=True, exist_ok=True)


def get_file_hash(file_path):
    """Return md5 hash of a file for duplicate detection."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def count_images_in_dataset(dataset_path):
    """Print number of image files per class."""
    print(f"\nClass counts in: {dataset_path}")
    total = 0
    if not dataset_path.exists():
        print("Folder does not exist.")
        return

    for class_dir in dataset_path.iterdir():
        if class_dir.is_dir():
            count = len([
                f for f in class_dir.iterdir()
                if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
            ])
            total += count
            print(f"  {class_dir.name}: {count}")
    print(f"  Total: {total}")


def safe_copy(src, dst):
    """Copy file safely, renaming if needed."""
    dst.parent.mkdir(parents=True, exist_ok=True)

    if not dst.exists():
        shutil.copy2(src, dst)
        return dst

    stem = dst.stem
    suffix = dst.suffix
    parent = dst.parent

    counter = 1
    while True:
        new_dst = parent / f"{stem}_{counter}{suffix}"
        if not new_dst.exists():
            shutil.copy2(src, new_dst)
            return new_dst
        counter += 1


# =========================
# MAIN CLEANING LOGIC
# =========================

def clean_dataset():
    if not SOURCE_DATASET.exists():
        print(f"ERROR: Source dataset not found: {SOURCE_DATASET.resolve()}")
        return

    create_folders()

    print("Starting dataset cleaning...")
    print(f"Source:  {SOURCE_DATASET.resolve()}")
    print(f"Cleaned: {CLEANED_DATASET.resolve()}")
    print(f"Review:  {REVIEW_DATASET.resolve()}")

    count_images_in_dataset(SOURCE_DATASET)

    seen_hashes = set()

    stats = {
        "total_files_seen": 0,
        "non_image_skipped": 0,
        "corrupted_skipped": 0,
        "duplicates_skipped": 0,
        "small_moved_to_review": 0,
        "cleaned_copied": 0
    }

    for class_dir in SOURCE_DATASET.iterdir():
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        print(f"\nProcessing class: {class_name}")

        for file_path in class_dir.iterdir():
            if not file_path.is_file():
                continue

            stats["total_files_seen"] += 1
            ext = file_path.suffix.lower()

            # 1. Skip non-image files
            if ext not in VALID_EXTENSIONS:
                stats["non_image_skipped"] += 1
                print(f"  [SKIP NON-IMAGE] {file_path.name}")
                continue

            # 2. Check corrupted images
            try:
                with Image.open(file_path) as img:
                    img.verify()
            except (UnidentifiedImageError, OSError, Exception):
                stats["corrupted_skipped"] += 1
                print(f"  [SKIP CORRUPTED] {file_path.name}")
                continue

            # Re-open after verify because verify() closes file state
            try:
                with Image.open(file_path) as img:
                    width, height = img.size
            except Exception:
                stats["corrupted_skipped"] += 1
                print(f"  [SKIP CORRUPTED AFTER VERIFY] {file_path.name}")
                continue

            # 3. Remove exact duplicates
            if REMOVE_DUPLICATES:
                try:
                    file_hash = get_file_hash(file_path)
                    if file_hash in seen_hashes:
                        stats["duplicates_skipped"] += 1
                        print(f"  [SKIP DUPLICATE] {file_path.name}")
                        continue
                    seen_hashes.add(file_hash)
                except Exception:
                    print(f"  [HASH ERROR, SKIPPING] {file_path.name}")
                    continue

            # 4. Move very small images to review folder
            if width < MIN_WIDTH or height < MIN_HEIGHT:
                dst = REVIEW_DATASET / class_name / file_path.name
                safe_copy(file_path, dst)
                stats["small_moved_to_review"] += 1
                print(f"  [MOVE TO REVIEW - SMALL IMAGE] {file_path.name} ({width}x{height})")
                continue

            # 5. Copy valid images to cleaned dataset
            dst = CLEANED_DATASET / class_name / file_path.name
            safe_copy(file_path, dst)
            stats["cleaned_copied"] += 1

    print("\nCleaning complete.")
    print("\nSummary:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    count_images_in_dataset(CLEANED_DATASET)
    count_images_in_dataset(REVIEW_DATASET)

    print("\nNext manual step:")
    print("1. Open the review_dataset folder and inspect small/suspicious images.")
    print("2. Delete truly bad images.")
    print("3. If some review images are usable, move them manually into cleaned_dataset.")
    print("4. Then use cleaned_dataset for train/val/test splitting.")


if __name__ == "__main__":
    clean_dataset()