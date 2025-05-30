import os
import shutil
import csv
import random
from pathlib import Path
from PIL import Image

# Configurare
input_root = r"data/Brain Tumor Segmentation Dataset/image"  # Calea catre datele brute (imagini)
mask_root = r"data/Brain Tumor Segmentation Dataset/mask"  # Calea catre mati
output_root = r"data/masks"
split_ratio = [0.8, 0.2]  # 80% train, 20% val
labeled_ratio = 1           # Cate din imagini din train sunt etichetate

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
    for sub in ["train/labeled/images", "train/labeled/masks", "train/unlabeled/images",
                "val/images", "val/masks"]:
        os.makedirs(os.path.join(output_root, sub), exist_ok=True)
        print(f"Created directory: {os.path.join(output_root, sub)}")

# Incarcare imagini si masti
def load_images():
    all_images = []
    for class_name, label in label_map.items():
        class_path = os.path.join(input_root, class_name).replace("\\", "/")
        print(f"Checking if the path exists: {os.path.exists(class_path)}")
        for fname in os.listdir(class_path):
            full_img_path = os.path.join(class_path, fname)

            # Cauta masca corespunzătoare în folderul de masti
            # Inlocuire .jpg cu _m.jpg pentru masca corespunzătoare
            mask_fname = fname.replace(".jpg", "_m.jpg")
            full_mask_path = os.path.join(mask_root, class_name, mask_fname)

            if os.path.exists(full_mask_path):  # Verifica daca exista masca corespunzatoare
                print(f"Found image and mask: {fname}")
                all_images.append((full_img_path, full_mask_path, label))
            else:
                print(f"Mask not found for image: {fname}")

    random.shuffle(all_images)  # Amestecăm imaginile
    return all_images



# Impartire în train/val
def split_dataset(images):
    total = len(images)
    n_train = int(total * split_ratio[0])
    return images[:n_train], images[n_train:]

# Salvare imagini și masti + CSV-urile pentru etichete
def save_images(split_name, entries, start_idx=0, labeled=False, csv_writer=None):
    for i, (img_path, mask_path, label) in enumerate(entries):
        # Returneaza numele fisierului original (fără cale)
        fname = os.path.basename(img_path)  # Numele original al fisierului

        # Redimensionare imagine la 224x224
        img = Image.open(img_path).convert("RGB").resize((224, 224))

        if split_name == "train":
            if labeled:
                img_save_path = os.path.join(output_root, "train/labeled/images", fname)
                mask_save_path = os.path.join(output_root, "train/labeled/masks", fname)
                if csv_writer:
                    csv_writer.writerow([fname, label])
            else:
                img_save_path = os.path.join(output_root, "train/unlabeled/images", fname)
                mask_save_path = None  # Fără mască pentru imaginile neetichetate
        else:
            img_save_path = os.path.join(output_root, f"{split_name}/images", fname)
            mask_save_path = os.path.join(output_root, f"{split_name}/masks", fname)
            if csv_writer:
                csv_writer.writerow([fname, label])

        # Salvare imagine redimensionata
        img.save(img_save_path)

        if mask_save_path:
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert("L").resize((224, 224))
                mask.save(mask_save_path)
            else:
                print(f"Mask not found: {mask_path}")

# Executare proces
make_dirs()
images = load_images()  # Citire imagini si masti
train_set, val_set = split_dataset(images)  # Doar train si val

# Etichetare si salvare setul de train
train_labeled_cutoff = int(len(train_set) * labeled_ratio)
train_labeled = train_set[:train_labeled_cutoff]
train_unlabeled = train_set[train_labeled_cutoff:]

# CSV pentru train
with open(os.path.join(output_root, "train_labels.csv"), "w", newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=';')
    writer.writerow(["image", "label"])
    save_images("train", train_labeled, labeled=True, csv_writer=writer)
    save_images("train", train_unlabeled, labeled=False, csv_writer=writer)

# CSV pentru val
with open(os.path.join(output_root, "val_labels.csv"), "w", newline='') as valfile:
    writer = csv.writer(valfile, delimiter=';')
    writer.writerow(["image", "label"])
    save_images("val", val_set, csv_writer=writer)

print("Datasetul a fost structurat cu succes in train/labeled, train/unlabeled și val!")
