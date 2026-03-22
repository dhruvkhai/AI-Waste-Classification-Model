import random
import shutil
from pathlib import Path

# =========================
# CONFIGURATION
# =========================

# Source cleaned dataset
SOURCE_DATASET = Path("../cleaned_dataset")

# Output split dataset
OUTPUT_DATASET = Path("../dataset")

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Random seed for reproducibility
RANDOM_SEED = 42

# Valid image extensions
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# =========================
# HELPER FUNCTIONS
# =========================

def create_output_folders(class_names):
    """Create train/val/test folders with class subfolders."""
    for split in ["train", "val", "test"]:
        for class_name in class_names:
            (OUTPUT_DATASET / split / class_name).mkdir(parents=True, exist_ok=True)

def safe_copy(src, dst):
    """Copy file safely, renaming if needed."""
    if not dst.exists():
        shutil.copy2(src, dst)
        return

    stem = dst.stem
    suffix = dst.suffix
    parent = dst.parent
    counter = 1

    while True:
        new_dst = parent / f"{stem}_{counter}{suffix}"
        if not new_dst.exists():
            shutil.copy2(src, new_dst)
            return
        counter += 1

def count_files(folder):
    """Count valid image files in a folder recursively."""
    return len([
        f for f in folder.rglob("*")
        if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
    ])

# =========================
# MAIN SPLITTING LOGIC
# =========================

def split_dataset():
    if not SOURCE_DATASET.exists():
        print(f"ERROR: Source dataset not found: {SOURCE_DATASET.resolve()}")
        return

    if abs((TRAIN_RATIO + VAL_RATIO + TEST_RATIO) - 1.0) > 1e-6:
        print("ERROR: Train/Val/Test ratios must add up to 1.0")
        return

    random.seed(RANDOM_SEED)

    class_dirs = [d for d in SOURCE_DATASET.iterdir() if d.is_dir()]
    class_names = [d.name for d in class_dirs]

    if not class_names:
        print("ERROR: No class folders found in cleaned_dataset.")
        return

    create_output_folders(class_names)

    print("Starting dataset split...")
    print(f"Source: {SOURCE_DATASET.resolve()}")
    print(f"Output: {OUTPUT_DATASET.resolve()}")
    print(f"Ratios -> Train: {TRAIN_RATIO}, Val: {VAL_RATIO}, Test: {TEST_RATIO}")

    total_train = total_val = total_test = 0

    for class_dir in class_dirs:
        class_name = class_dir.name
        images = [
            f for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
        ]

        random.shuffle(images)

        total_images = len(images)
        train_count = int(total_images * TRAIN_RATIO)
        val_count = int(total_images * VAL_RATIO)
        test_count = total_images - train_count - val_count

        train_files = images[:train_count]
        val_files = images[train_count:train_count + val_count]
        test_files = images[train_count + val_count:]

        for file_path in train_files:
            dst = OUTPUT_DATASET / "train" / class_name / file_path.name
            safe_copy(file_path, dst)

        for file_path in val_files:
            dst = OUTPUT_DATASET / "val" / class_name / file_path.name
            safe_copy(file_path, dst)

        for file_path in test_files:
            dst = OUTPUT_DATASET / "test" / class_name / file_path.name
            safe_copy(file_path, dst)

        total_train += len(train_files)
        total_val += len(val_files)
        total_test += len(test_files)

        print(f"\nClass: {class_name}")
        print(f"  Total: {total_images}")
        print(f"  Train: {len(train_files)}")
        print(f"  Val:   {len(val_files)}")
        print(f"  Test:  {len(test_files)}")

    print("\nSplit complete.")
    print("\nFinal totals:")
    print(f"  Train: {total_train}")
    print(f"  Val:   {total_val}")
    print(f"  Test:  {total_test}")
    print(f"  Total: {total_train + total_val + total_test}")

    print("\nVerification counts:")
    print(f"  Train folder count: {count_files(OUTPUT_DATASET / 'train')}")
    print(f"  Val folder count:   {count_files(OUTPUT_DATASET / 'val')}")
    print(f"  Test folder count:  {count_files(OUTPUT_DATASET / 'test')}")

if __name__ == "__main__":
    split_dataset()
    