# TrainTeacher.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
from tqdm import tqdm
from CNNmodels.my_teacher_CNN import TeacherModel
# === Config ===
IMAGE_DIR = 'data/labels/train/labeled'
LABELS_CSV = 'data/labels/labels.csv'
MODEL_PATH = 'teacher.pth'
BATCH_SIZE = 32
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dacă GPU-ul este disponibil, afiseaza detalii despre GPU
if torch.cuda.is_available():
    print(f"GPU-ul disponibil: {torch.cuda.get_device_name(0)}")
    print(f"Memorie totală GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**2} MB")
else:
    print("GPU nu este disponibil, se foloseste CPU.")

# === Dataset Loader ===
class LabeledDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.data = pd.read_csv(csv_path, delimiter =";")
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]
        #img_path = os.path.join(self.img_dir, img_name)
        img_path = os.path.abspath(os.path.join(self.img_dir, img_name))  # calea absoluta
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

# === Transformari ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

# === Dataset & Dataloader ===
dataset = LabeledDataset(LABELS_CSV, IMAGE_DIR, transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# === Model ===
model = TeacherModel(num_classes=4, lr=1e-4, device=DEVICE)
# model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)  # Inlocuieste pretrained cu weights
# model.fc = nn.Linear(model.fc.in_features, 4)  # 4 clase: benign / malign /pituitar/ no tumor
# model = model.to(DEVICE)


# === Antrenare ===
model.train(dataloader, epochs=EPOCHS)

# === Salvare model ===
model.save(MODEL_PATH)
