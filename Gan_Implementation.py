import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

# Hyperparameters
z_dim = 128
lr = 0.0002
batch_size = 128
epochs = 50
image_size = 64
channels_img = 3
features_gen = 128  # Increased features
features_disc = 128  # Increased features
num_classes = 10
device = "cuda" if torch.cuda.is_available() else "cpu"

# Data Preparation
transform = transforms.Compose([  # compose fit krdeta hai saare transformations ko single pipeline me
    transforms.Resize(image_size),
    transforms.RandomHorizontalFlip(),  #yaha pe hojaegi flip 50% chance ke saath , taaki dataset ka variabiliy increase ho , data augmention , prevents overfitting and improves generalization
    transforms.ToTensor(), #tensor format me badaldiya
    transforms.Normalize([0.5] * 3, [0.5] * 3)# mean hai ([0.5],[0.5],[0.5]) , standard deviation hai ([0.5],[0.5],[0.5]), so , The pixel values are normalized to the range [-1, 1] by subtracting 0.5 and dividing by 0.5 for each channel.
])

dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

def show(tensor, num=25):
    grid = make_grid(tensor[:num], nrow=5, normalize=True).permute(1, 2, 0)
    plt.imshow(grid.cpu().numpy())
    plt.show()

# Generator
class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_gen):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            self._block(z_dim, features_gen * 8, 4, 1, 0),
            self._block(features_gen * 8, features_gen * 4, 4, 2, 1),
            self._block(features_gen * 4, features_gen * 2, 4, 2, 1),
            self._block(features_gen * 2, features_gen, 4, 2, 1),
            nn.ConvTranspose2d(features_gen, channels_img, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, channels_img, features_disc):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(channels_img, features_disc, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            self._block(features_disc, features_disc * 2, 4, 2, 1),
            self._block(features_disc * 2, features_disc * 4, 4, 2, 1),
            self._block(features_disc * 4, features_disc * 8, 4, 2, 1),
            nn.Flatten(),
        )

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(features_disc * 8 * (image_size // 16) * (image_size // 16), num_classes),
        )

        self.adv_head = nn.Sequential(
            nn.Linear(features_disc * 8 * (image_size // 16) * (image_size // 16), 1),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        features = self.disc(x)
        features_flattened = features.view(features.size(0), -1)
        classification_logits = self.classifier(features_flattened)
        adversarial_logits = self.adv_head(features_flattened)
        return adversarial_logits, classification_logits

# Initialize models
gen = Generator(z_dim, channels_img, features_gen).to(device)
disc = Discriminator(channels_img, features_disc).to(device)

# Optimizers and Loss
opt_gen = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
opt_disc = torch.optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))
loss_fn = nn.BCEWithLogitsLoss()
cls_loss_fn = nn.CrossEntropyLoss()

# Training Loop
for epoch in range(epochs):
    for batch_idx, (real, labels) in enumerate(tqdm(dataloader)):
        real = real.to(device)
        labels = labels.to(device)
        cur_batch_size = real.size(0)

        # Train Discriminator
        noise = torch.randn(cur_batch_size, z_dim, 1, 1).to(device)
        fake = gen(noise)

        real_disc, real_cls = disc(real)
        fake_disc, _ = disc(fake.detach())

        real_loss = loss_fn(real_disc, torch.ones_like(real_disc))
        fake_loss = loss_fn(fake_disc, torch.zeros_like(fake_disc))
        cls_loss = cls_loss_fn(real_cls, labels)
        disc_loss = (real_loss + fake_loss) / 2 + cls_loss

        disc.zero_grad()  # clear previous gradients
        disc_loss.backward()  # backpropagate the loss
        opt_disc.step() # Update discriminator weights

        # Train Generator
        fake_disc, _ = disc(fake)
        gen_loss = loss_fn(fake_disc, torch.ones_like(fake_disc))

        gen.zero_grad()
        gen_loss.backward()
        opt_gen.step()

    print(f"Epoch [{epoch}/{epochs}] | D Loss: {disc_loss:.4f} | G Loss: {gen_loss:.4f}")

# Generate and Show Samples
noise = torch.randn(25, z_dim, 1, 1).to(device)
fake_images = gen(noise)
show(fake_images)
