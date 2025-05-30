import os
import shutil
import csv
import random
from pathlib import Path
from PIL import Image

# Config
input_root = r"data/Kaggle/Testing/"  # path către datele brute (Kaggle)
output_root = r"data/labels"
split_ratio = [0.8, 0.2]  # train, val
labeled_ratio = 0.6            # cate din imaginile de train sunt etichetate

# Mapare clase la cod numeric
label_map = {
    "glioma": 0,
    "meningioma": 1,
    "pituitary": 2,
    "notumor": 3
}

# Creare directoare
def make_dirs():
    os.makedirs(output_root, exist_ok=True)
    for sub in ["train/labeled", "train/unlabeled", "val"]:
        os.makedirs(os.path.join(output_root, sub), exist_ok=True)
        print(os.path.join(output_root, sub))

# Incarcara imaginilor
def load_images():
    all_images = []
    for class_name, label in label_map.items():
        class_path = os.path.join(input_root, class_name).replace("\\", "/")
        print(f"Checking if the path exists: {os.path.exists(class_path)}")
        print(class_path + " test")
        for fname in os.listdir(class_path):
            full_path = os.path.join(class_path, fname)
            all_images.append((full_path, label))
    random.shuffle(all_images)
    return all_images

# Impartire in train/val/
def split_dataset(images):
    total = len(images)
    n_train = int(total * split_ratio[0])
    n_val = int(total * split_ratio[1])
    return images[:n_train], images[n_train:n_train+n_val], images[n_train+n_val:]

# Salvăm imaginile + labels.csv
def save_images(split_name, entries, start_idx=0, labeled=False, csv_writer=None):
    for i, (img_path, label) in enumerate(entries):
        img = Image.open(img_path).convert("RGB").resize((224, 224))
        #fname = f"{split_name}_{i+start_idx:04d}.png"
        fname = os.path.basename(img_path)  # Numele original al fișierului

        if split_name == "train":
            if labeled:
                save_path = os.path.join(output_root, "train/labeled", fname)
                if csv_writer:
                    csv_writer.writerow([fname, label])  # Adauga 2 coloane
            else:
                save_path = os.path.join(output_root, "train/unlabeled", fname)
        else:
            save_path = os.path.join(output_root, split_name, fname)
            if csv_writer:
                csv_writer.writerow([fname, label])  # Adauga 2 coloane

        img.save(save_path)

# Executăm
make_dirs()
images = load_images()
train_set, val_set, test_set = split_dataset(images)

# Etichetare și salvare train set
train_labeled_cutoff = int(len(train_set) * labeled_ratio)
train_labeled = train_set[:train_labeled_cutoff]
train_unlabeled = train_set[train_labeled_cutoff:]

with open(os.path.join(output_root, "labels.csv"), "w", newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=';')  # Punct is virgula pentru separator
    writer.writerow(["image", "label"])  # Titlurile pentru coloane
    save_images("train", train_labeled, labeled=True, csv_writer=writer)
    save_images("train", train_unlabeled, labeled=False, csv_writer=writer)

# Salvare CSV pentru val și test
with open(os.path.join(output_root, "val_labels.csv"), "w", newline='') as valfile:
    writer = csv.writer(valfile, delimiter=';')  # Punct is virgula pentru separator
    writer.writerow(["image", "label"])  # Titlurile pentru coloane
    save_images("val", val_set, csv_writer=writer)

# with open(os.path.join(output_root, "test_labels.csv"), "w", newline='') as testfile:
#     writer = csv.writer(testfile, delimiter=';')  # Punct is virgula pentru separator
#     writer.writerow(["image", "label"])  # Titlurile pentru coloane
#     save_images("test", test_set, csv_writer=writer)

print("Dataset structurat cu succes în train/labeled, train/unlabeled, val și test!")
