import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Transformation des images (normalisation et redimensionnement)
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Taille standard pour l'entraînement
    transforms.ToTensor(),       # Conversion en tenseur
    transforms.Normalize((0.5,), (0.5,))  # Normalisation pour faciliter l'entraînement
])

# Chargement d'un dataset (FashionMNIST en exemple)
dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)

# DataLoader pour gérer les lots pendant l'entraînement
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

print(f"Taille du dataset : {len(dataset)} images")
