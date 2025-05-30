import torch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import numpy as np
from torchvision import models
from PIL import Image
from torchvision import transforms
import os
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from CNNmodels.my_student_CNN import StudentModelResNet18
from CNNmodels.my_teacher_CNN import TeacherModel


#model = models.resnet18(pretrained=True)  # ResNet50
#model.fc = torch.nn.Linear(model.fc.in_features, 4)  # 4 clase: benign, malign, pituitar, fără tumora

# Setez dispozitivul, daca CUDA este disponibi, folosesc GPU altfel CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

student = StudentModelResNet18(n_classes=4, lr=1e-4, device=DEVICE) # 4 clase: benign, malign, pituitar, fără
student.load('student.pth', DEVICE)

#Incarc modelul salvat
# student = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
# student.fc = torch.nn.Linear(student.fc.in_features, 4)
# student.load_state_dict(torch.load('student.pth'), strict=False)
# student.eval()
# student = student.to(DEVICE)

#student = TeacherModel(num_classes=4, lr=1e-4, device=DEVICE)

#student.model.eval()  # Setează modelul în modul de inferență


# Transformari pentru imagini
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# Citirea etichetelor din CSV
def load_test_labels(csv_path):
    df = pd.read_csv(csv_path, sep=';')
    # Presupunem că CSV-ul conține doua coloane: 'image' (numele imaginii) si 'label' (eticheta reala)
    #labels_dict = {row['image']: row['label'] for _, row in df.iterrows()}
    labels_dict = {row['image']: int(row['label']) for _, row in df.iterrows()}
    return labels_dict


# Definirea claselor
classes = ['Benign', 'Malign', 'Pituitar', 'No Tumor']


# Funcția pentru a face predicția pentru o imagine
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Adaugare dimensiune suplimentara pentru batch
    image = image.to(DEVICE)

    with torch.no_grad():  # Dezactivarea calculul gradientului pentru infereta
        output = student.model(image)

    # Obține probabilitățile pentru fiecare clasa
    probabilities = torch.softmax(output, dim=1)
    predicted_class_idx = torch.argmax(probabilities).item()  # Clasa prezisă

    # Afisare probabilitati pentru fiecare clasă
    print(f"Imaginea: {image_path}")
    print(f"Probabilitatile pentru fiecare clasă:")
    for i, prob in enumerate(probabilities[0]):
        print(f"Clasa {classes[i]}: {prob * 100:.2f}%")

    print(f"Clasa prezisă: {classes[predicted_class_idx]}")
    print("-" * 50)

    return predicted_class_idx


# Parcurge toate fisierele din folderul de test
test_folder = 'data/labels/val'  # Calea catre folderul de test
test_labels_csv = 'data/labels/val_labels.csv'  # Calea catre fișierul CSV-ul cu etichetele reale
y_true = []  # Etichetele reale
y_pred = []  # Predictiile modelului

# Incarca etichetele reale din CSV
real_labels = load_test_labels(test_labels_csv)

# Parcurge imagini si face predictia
for image_name in os.listdir(test_folder):
    if image_name.endswith('.png') or image_name.endswith('.jpg'):  # Verifica tipul fisierului
        image_path = os.path.join(test_folder, image_name)

        # Obine eticheta reala din dictionarul incarcat din CSV
        real_label = real_labels.get(image_name)  # Se presupune ca numele imaginii exista si este in dictionar

        if real_label is not None:
            # Face predicția
            predicted_class = predict_image(image_path)

            # Adauga eticheta reala si predictia
            y_true.append(real_label)
            y_pred.append(predicted_class)


# Calcularea metricilor de performanta
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')  # Pentru mai multe clase
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')
cm = confusion_matrix(y_true, y_pred)

print("y_true:", y_true[:50])
print("y_pred:", y_pred[:50])

# Afisare rezultate
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1-Score: {f1 * 100:.2f}%")
print("Confusion Matrix:")
print(cm)

# Vizualizare matrice de confuzie
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title("Matricea de Confuzie")
plt.xlabel("Clasa prezisă")
plt.ylabel("Clasa reală")
plt.show()
