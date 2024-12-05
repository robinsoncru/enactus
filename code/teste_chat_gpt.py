import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Transformation des images
def get_dataloader(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalisation pour niveaux de gris
    ])

    # Charger FashionMNIST
    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    return DataLoader(trainset, batch_size=batch_size, shuffle=True)

# Définir le Générateur
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 1 * 28 * 28)  # 1 canal pour image 28x28 (niveaux de gris)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, z):
        z = self.relu(self.fc1(z))
        z = self.relu(self.fc2(z))
        z = self.relu(self.fc3(z))
        z = self.tanh(self.fc4(z))
        return z.view(-1, 1, 28, 28)  # Redimensionner en image 28x28

# Définir le Discriminateur
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(1 * 28 * 28, 1024)  # Entrée : image 28x28x1
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)  # Sortie : probabilité réelle (1) ou générée (0)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img):
        img = img.view(-1, 1 * 28 * 28)  # Redimensionner l'image
        img = self.leaky_relu(self.fc1(img))
        img = self.leaky_relu(self.fc2(img))
        img = self.leaky_relu(self.fc3(img))
        img = self.sigmoid(self.fc4(img))
        return img

# Initialisation du GAN
def initialize_gan():
    generator = Generator().cuda()
    discriminator = Discriminator().cuda()
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    return generator, discriminator, criterion, optimizer_g, optimizer_d

# Entraîner le GAN
def train_gan(generator, discriminator, criterion, optimizer_g, optimizer_d, trainloader, num_epochs=100):
    for epoch in range(num_epochs):
        for i, (imgs, _) in enumerate(trainloader):
            # Préparer les données réelles
            real_imgs = imgs.cuda()
            batch_size = real_imgs.size(0)
            labels_real = torch.ones(batch_size, 1).cuda()
            labels_fake = torch.zeros(batch_size, 1).cuda()

            # -----------------
            # Entraîner le Discriminateur
            # -----------------
            optimizer_d.zero_grad()

            # Perte sur les images réelles
            output_real = discriminator(real_imgs)
            loss_real = criterion(output_real, labels_real)

            # Perte sur les images générées
            z = torch.randn(batch_size, 100).cuda()
            fake_imgs = generator(z)
            output_fake = discriminator(fake_imgs.detach())
            loss_fake = criterion(output_fake, labels_fake)

            # Perte totale du discriminateur
            loss_d = loss_real + loss_fake
            loss_d.backward()
            optimizer_d.step()

            # -----------------
            # Entraîner le Générateur
            # -----------------
            optimizer_g.zero_grad()
            output_fake = discriminator(fake_imgs)
            loss_g = criterion(output_fake, labels_real)  # Tromper le discriminateur
            loss_g.backward()
            optimizer_g.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss D: {loss_d.item()}, Loss G: {loss_g.item()}")

        # Afficher des images générées toutes les 10 époques
        if (epoch + 1) % 50 == 0:
            show_generated_images(fake_imgs)

# Affichage des images générées
def show_generated_images(images):
    images = images.cpu().detach().numpy()
    images = images.transpose(0, 2, 3, 1).squeeze()  # Convertir en (batch_size, 28, 28)
    plt.figure(figsize=(10, 10))
    for i in range(10):
        plt.subplot(1, 10, i + 1)
        plt.imshow(images[i] * 0.5 + 0.5, cmap='gray')  # Re-scaler de [-1, 1] à [0, 1]
        plt.axis('off')
    plt.show()

def download_discr_gene(generator, discriminator):
    # Enregistrer les poids
    torch.save(generator.state_dict(), 'generator_weights.pth')
    torch.save(discriminator.state_dict(), 'discriminator_weights.pth')

def load_discri_gene():
    # Charger les poids
    generator.load_state_dict(torch.load('generator_weights.pth'))
    discriminator.load_state_dict(torch.load('discriminator_weights.pth'))
    return generator, discriminator


# Lancer l'entraînement
trainloader = get_dataloader()
generator, discriminator, criterion, optimizer_g, optimizer_d = initialize_gan()
train_gan(generator, discriminator, criterion, optimizer_g, optimizer_d, trainloader)
download_discr_gene(generator, discriminator)