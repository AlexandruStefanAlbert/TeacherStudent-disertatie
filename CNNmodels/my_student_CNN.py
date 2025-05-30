import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.optim as optim
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# CNNmodels/my_student_CNN.py

class StudentModelResNet18:
    def __init__(self, n_classes=4, lr=1e-4, device=None):
        #super(StudentModelResNet18, self).__init__()

        #self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, n_classes)
        self.model.to(self.device)

        # === Optimizare ===
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def forward(self, x):
        return self.model(x)

    def train_model(self, dataloader, epochs=20):
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"[Epoch {epoch+1}] Loss: {total_loss / len(dataloader):.4f}")

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"\nModel salvat ca: {path}")

    def load(self, path, device):
        self.model.load_state_dict(torch.load(path, map_location=device))
        self.model.to(device)
        self.model.eval()
        print(f"Model încărcat de la: {path}")

