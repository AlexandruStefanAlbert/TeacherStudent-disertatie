# TrainStudent.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from tqdm import tqdm
from CNNmodels.my_student_CNN import StudentModelResNet18

# === Config ===
LABELED_CSV = 'data/labels/labels.csv'
PSEUDO_CSV = 'data/labels/pseudo_labels.csv'
LABELED_DIR = 'data/labels/train/labeled'
UNLABELED_DIR = 'data/labels/train/unlabeled'
MODEL_PATH = 'student.pth'
BATCH_SIZE = 32
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dacă GPU-ul este disponibil, utilizeze GPU pentru antrenare + afisare date GPU
if torch.cuda.is_available():
    print(f"GPU-ul disponibil: {torch.cuda.get_device_name(0)}")
    print(f"Memorie totala GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**2} MB")
else:
    print("GPU nu este disponibil, se foloseste CPU.")

# === Dataset combinat ===
class CombinedDataset(Dataset):
    def __init__(self, labeled_csv, pseudo_csv, labeled_dir, unlabeled_dir, transform=None):
        self.transform = transform
        self.entries = []

        # imagini etichetate
        df_labeled = pd.read_csv(labeled_csv, sep=';')
        #print(df_labeled.head())
        for _, row in df_labeled.iterrows():
            self.entries.append((os.path.join(labeled_dir, row['image']), row['label']))

        # imagini pseudo-etichetate
        df_pseudo = pd.read_csv(pseudo_csv, sep=';')
        #print(df_labeled.head())
        for _, row in df_pseudo.iterrows():
            self.entries.append((os.path.join(unlabeled_dir, row['image']), row['pseudo_label']))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        path, label = self.entries[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, int(label)



# === Transformări ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])


# === Dataloader ===
dataset = CombinedDataset(LABELED_CSV, PSEUDO_CSV, LABELED_DIR, UNLABELED_DIR, transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
print(dataset.__len__())

# === Model Student (mai simplu) ===
#model = models.resnet18(pretrained=True)
#model.fc = nn.Linear(model.fc.in_features, 4) #4 clase: benign, malign, pituitar, no tumor
#model = StudentModel(n_classes=4)  # 4 clase: benign, malign, pituitar, no tumor
#model = model.to(DEVICE)
model = StudentModelResNet18(n_classes=4, lr=1e-4, device=DEVICE)


# === Antrenare ===
model.train_model(dataloader, epochs=EPOCHS)

# === Salvare student ===
model.save(MODEL_PATH)
