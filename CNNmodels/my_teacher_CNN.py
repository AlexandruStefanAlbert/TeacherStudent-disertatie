# teacher_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm

class TeacherModel:
    def __init__(self, num_classes=4, lr=1e-4, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model.to(self.device)

        # === Optimizare ===
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, dataloader, epochs=20):
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

            avg_loss = total_loss / len(dataloader)
            print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"\nModel salvat ca: {path}")

    def load(self, path, device):
        self.model.load_state_dict(torch.load(path, map_location=device))
        self.model.to(device)
        self.model.eval()
        print(f"Model încărcat de la: {path}")
