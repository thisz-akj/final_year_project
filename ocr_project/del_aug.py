import os

def cleanup_dataset(dataset_dir):
    for class_name in os.listdir(dataset_dir):
        class_path = os.path.join(dataset_dir, class_name)

        if not os.path.isdir(class_path):
            continue

        for file in os.listdir(class_path):
            if "_" in file:   # matches ka_001.png etc.
                os.remove(os.path.join(class_path, file))

        print(f"[CLEANED] {class_name}")

if __name__ == "__main__":
    cleanup_dataset("/home/workstation/Documents/Azadec22b1109/ocr_project/data")
