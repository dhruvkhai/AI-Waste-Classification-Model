import os
import glob

# Paths to label directories
train_labels_dir = "archive/Warp-D/train/labels"
test_labels_dir = "archive/Warp-D/test/labels"
label_dirs = [train_labels_dir, test_labels_dir]

def remap_labels_to_class(directory, target_class=1):
    txt_files = glob.glob(os.path.join(directory, "*.txt"))
    modified_count = 0
    for txt_file in txt_files:
        if not os.path.isfile(txt_file):
            continue
        # Only rewrite if there are lines
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                # YOLO format: class x_center y_center width height
                parts[0] = str(target_class)
                new_line = " ".join(parts) + "\n"
                new_lines.append(new_line)
            else:
                # keep empty or malformed lines as is
                new_lines.append(line)
        
        # Write back if lines exist
        if lines:
            with open(txt_file, 'w') as f:
                f.writelines(new_lines)
            modified_count += 1
            
    return modified_count

total_modified = 0
for d in label_dirs:
    count = remap_labels_to_class(d, target_class=1)
    print(f"Directory: {d} | Modified {count} files.")
    total_modified += count
    
print(f"Total files remapped to class 1: {total_modified}")
