import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import vgg19
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = 64
batch_size = 16
lr = 1e-4
num_epochs = 12

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),  # Normalize to [-1, 1]
])

dataset = datasets.LFWPeople(root="./data", split="train", transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.upsample = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU(),
            *[self._residual_block(64) for _ in range(5)],  # Residual blocks
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),  # Upsampling by factor of 2
            nn.PReLU(),
            nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4),
            nn.Tanh()
        )

    def _residual_block(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return self.upsample(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            self._block(3, 64, 3, 1, 1, False),
            self._block(64, 64, 3, 2, 1, True),
            self._block(64, 128, 3, 1, 1, True),
            self._block(128, 128, 3, 2, 1, True),
            self._block(128, 256, 3, 1, 1, True),
            self._block(256, 256, 3, 2, 1, True),
            nn.Flatten(),
            nn.Linear(256 * (image_size // 8) * (image_size // 8), 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding, batch_norm):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

gen = Generator().to(device)
disc = Discriminator().to(device)

adversarial_loss = nn.BCELoss()
content_loss = nn.MSELoss()

optimizer_G = torch.optim.Adam(gen.parameters(), lr=lr)
optimizer_D = torch.optim.Adam(disc.parameters(), lr=lr)

vgg = vgg19(pretrained=True).features[:36].eval().to(device)
for param in vgg.parameters():
    param.requires_grad = False

for epoch in range(num_epochs):
    loop = tqdm(dataloader, leave=True)
    for idx, (hr_images, _) in enumerate(loop):
        hr_images = hr_images.to(device)

        # Create low-resolution images
        lr_images = nn.functional.interpolate(hr_images, scale_factor=0.5, mode="bicubic")

        # Train Discriminator
        real_labels = torch.ones(hr_images.size(0), 1).to(device)
        fake_labels = torch.zeros(hr_images.size(0), 1).to(device)

        fake_images = gen(lr_images)
        disc_real = disc(hr_images)
        disc_fake = disc(fake_images.detach())
        loss_D_real = adversarial_loss(disc_real, real_labels)
        loss_D_fake = adversarial_loss(disc_fake, fake_labels)
        loss_D = (loss_D_real + loss_D_fake) / 2

        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # Train Generator
        disc_fake = disc(fake_images)
        loss_G_adv = adversarial_loss(disc_fake, real_labels)
        loss_G_content = content_loss(vgg(fake_images), vgg(hr_images))
        loss_G = loss_G_content + 1e-3 * loss_G_adv

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        # Update progress bar
        loop.set_description(f"Epoch [{epoch + 1}/{num_epochs}]")
        loop.set_postfix(loss_D=loss_D.item(), loss_G=loss_G.item())

torch.save(gen.state_dict(), "srgan_generator.pth")
torch.save(disc.state_dict(), "srgan_discriminator.pth")

def visualize_results(gen, dataloader):
    gen.eval()
    with torch.no_grad():
        for lr_images, hr_images in dataloader:
            lr_images = lr_images.to(device)
            fake_images = gen(lr_images)
            vutils.save_image(fake_images, "output_samples.png", normalize=True)
            break

visualize_results(gen, dataloader)
