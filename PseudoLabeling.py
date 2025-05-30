# PseudoLabeling.py
import os
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from tqdm import tqdm
import numpy as np
from CNNmodels.my_teacher_CNN import TeacherModel

# === Config ===
UNLABELED_DIR = 'data/labels/train/unlabeled'
MODEL_PATH = 'teacher.pth'
OUTPUT_CSV = 'data/labels/pseudo_labels.csv'
TOP_K = 300     # imagini per clasa
TOP_P = 1       # scoruri pastrate per imagine => cel mai bun scor pentru o clasa
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Dataset fara etichete ===
class UnlabeledDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.image_names = os.listdir(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        #img_path = os.path.join(self.image_dir, img_name)
        img_path = os.path.abspath(os.path.join(self.img_dir, img_name)) #calea absoluta
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, img_name

# === Transform ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# === Load dataset ===
dataset = UnlabeledDataset(UNLABELED_DIR, transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# === Load Teacher model ===
# model = models.resnet50()
# model.fc = torch.nn.Linear(model.fc.in_features, 4) #4 clase: benign, malign, pituitar si no tumor
# model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
# model = model.to(DEVICE)
teacher = TeacherModel(num_classes=4, lr=1e-4, device=DEVICE)
teacher.model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
teacher.model.eval()

# === Predict ===
all_preds = []
all_names = []

with torch.no_grad():
    for images, names in tqdm(dataloader, desc="Predicting"):
        images = images.to(DEVICE)
        outputs = teacher.model(images)
        probs = F.softmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(probs)
        all_names.extend(names)

all_preds = np.array(all_preds)

# === Select Top-K imagini per clasa ===
selected = []
for cls in range(4):
    cls_probs = all_preds[:, cls]
    top_k_indices = cls_probs.argsort()[::-1][:TOP_K]
    for idx in top_k_indices:
        name = all_names[idx]
        scores = all_preds[idx]
        top_p_indices = scores.argsort()[::-1][:TOP_P]
        filtered_scores = np.zeros_like(scores)
        filtered_scores[top_p_indices] = scores[top_p_indices]
        label = filtered_scores.argmax()  # cea mai probabilă clasă
        selected.append((name, label, *filtered_scores))

# Un dict pentru a pastra cea mai buna eticheta per imagine
best_labels = {}

for name, _, glioma, meningioma, pituitary, notumor in selected:
    scores = np.array([glioma, meningioma, pituitary, notumor])
    max_score = scores.max()
    label = scores.argmax()

    if name not in best_labels or best_labels[name][0] < max_score:
        best_labels[name] = (max_score, label, glioma, meningioma, pituitary, notumor)

# Convertim dict-ul în lista pentru DataFrame
filtered_selected = [(name, v[1], v[2], v[3], v[4], v[5]) for name, v in best_labels.items()]

# === Salvare CSV ===
columns = ['image', 'pseudo_label', 'glioma', 'meningioma', 'pituitary', 'notumor']
df = pd.DataFrame(filtered_selected, columns=columns)
df.to_csv(OUTPUT_CSV, index=False, sep=';')
# df = pd.DataFrame(selected, columns=columns)
# df.to_csv(OUTPUT_CSV, index=False, sep=';')
print(f"\nTop-K Pseudo-labels salvate în: {OUTPUT_CSV}")
