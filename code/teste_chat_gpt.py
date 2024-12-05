import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Transformation des images (mise à l'échelle à [0, 1] et conversion en Tensor)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalisation pour les images RGB
])

# Charger la base de données CIFAR-10
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

# Vérifier le nombre d'images dans la base
print(f'Total number of images: {len(trainset)}')

import torch.nn as nn
import torch.optim as optim

# Définir le Générateur
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(100, 256)  # Entrée de 100 (vecteur latent)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 3 * 32 * 32)  # Sortie: 3 canaux pour une image de 32x32
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()  # Normalisation de l'image entre [-1, 1]

    def forward(self, z):
        z = self.relu(self.fc1(z))
        z = self.relu(self.fc2(z))
        z = self.relu(self.fc3(z))
        z = self.tanh(self.fc4(z))
        return z.view(-1, 3, 32, 32)  # Redimensionner en image 32x32

# Définir le Discriminateur
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 1024)  # Entrée: image de 32x32x3
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)  # Sortie: probabilité réelle (1) ou générée (0)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img):
        img = img.view(-1, 3 * 32 * 32)  # Redimensionner l'image
        img = self.leaky_relu(self.fc1(img))
        img = self.leaky_relu(self.fc2(img))
        img = self.leaky_relu(self.fc3(img))
        img = self.sigmoid(self.fc4(img))
        return img

# Initialiser le générateur et le discriminateur
generator = Generator().cuda()  # Déplacer sur GPU si disponible
discriminator = Discriminator().cuda()

# Définir les optimisateurs et la fonction de perte
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Fonction pour entraîner le GAN
num_epochs = 50
for epoch in range(num_epochs):
    for i, (imgs, _) in enumerate(trainloader):
        # Préparer les étiquettes pour la perte
        real_imgs = imgs.cuda()
        batch_size = real_imgs.size(0)
        labels_real = torch.ones(batch_size, 1).cuda()  # étiquettes réelles (1)
        labels_fake = torch.zeros(batch_size, 1).cuda()  # étiquettes générées (0)

        # -----------------
        # Entraînement du Discriminateur
        # -----------------

        # Discriminateur sur images réelles
        optimizer_d.zero_grad()
        output_real = discriminator(real_imgs)
        loss_real = criterion(output_real, labels_real)

        # Discriminateur sur images générées
        z = torch.randn(batch_size, 100).cuda()  # Bruit aléatoire pour le générateur
        fake_imgs = generator(z)
        output_fake = discriminator(fake_imgs.detach())  # Détacher pour ne pas entraîner le générateur
        loss_fake = criterion(output_fake, labels_fake)

        # Perte totale du discriminateur
        loss_d = loss_real + loss_fake
        loss_d.backward()
        optimizer_d.step()

        # -----------------
        # Entraînement du Générateur
        # -----------------

        optimizer_g.zero_grad()
        output_fake = discriminator(fake_imgs)
        loss_g = criterion(output_fake, labels_real)  # On veut tromper le discriminateur
        loss_g.backward()
        optimizer_g.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss D: {loss_d.item()}, Loss G: {loss_g.item()}")

    # Sauvegarder des images générées à chaque époque
    if (epoch + 1) % 10 == 0:
        import matplotlib.pyplot as plt
        plt.imshow(fake_imgs[0].cpu().detach().numpy().transpose(1, 2, 0) * 0.5 + 0.5)  # Afficher la première image générée
        plt.show()

import matplotlib.pyplot as plt

# Fonction pour afficher les images générées
def show_generated_images(images):
    images = images.cpu().detach().numpy()
    images = images.transpose(0, 2, 3, 1)  # Convertir en forme (batch_size, 32, 32, 3)
    plt.figure(figsize=(10, 10))
    for i in range(10):
        plt.subplot(1, 10, i + 1)
        plt.imshow(images[i] * 0.5 + 0.5)  # Re-scaler de [-1, 1] à [0, 1]
        plt.axis('off')
    plt.show()

# Après une époque d'entraînement, afficher les images générées
show_generated_images(fake_imgs)
