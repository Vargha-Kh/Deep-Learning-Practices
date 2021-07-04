import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import matplotlib.animation as animation
from IPython.display import HTML
import matplotlib.pyplot as plt
import numpy as np

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

epochs = 30
batch_size = 128
latent_dim = 100
image_size = 28
channel_size = 1
sample_interval = 400
gen_lr = 2e-4
disc_lr = 2e-4
decay = 6e-8
logdir = "/content/drive/MyDrive/TB"


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# Dataloader
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.MNIST(root='mnist_data',
                               train=True,
                               transform=transforms,
                               download=True)
# define the data loaders
dataloader = DataLoader(dataset=train_dataset,
                        batch_size=batch_size,
                        shuffle=True)


# Model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.l1 = nn.Sequential(
            nn.Linear(latent_dim, 256 * 7 * 7, bias=False),
            nn.BatchNorm1d(256 * 7 * 7),
            nn.ReLU()
        )
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, out_channels=256, kernel_size=11, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        # out = self.l1(z)
        # out = out.view((-1, 256, 7, 7))
        gen = self.conv_blocks(z)
        return gen


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=11, stride=1, padding=2, bias=True),
            nn.Sigmoid()
        )

        self.linear1 = nn.Linear(128 * 7 * 7, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img):
        out = self.model(img)
        # out = out.view((-1, 128 * 7 * 7))
        # linear = self.linear1(out)
        # dis = self.sigmoid(linear)
        return out


# Compiling Model
criterion = torch.nn.BCELoss()
generator = Generator()
discriminator = Discriminator()

generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)
if cuda:
    generator.cuda()
    discriminator.cuda()
    criterion.cuda()

optimizer_G = torch.optim.Adam(generator.parameters(), lr=gen_lr, weight_decay=decay, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=disc_lr, weight_decay=decay, betas=(0.5, 0.999))
reduce_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, mode='min', patience=3000, verbose=True)

# Training
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
writer = SummaryWriter(logdir)

img_list = []
G_losses = []
D_losses = []
iters = 0
real_label = 1.
fake_label = 0.
fixed_noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)

print("Starting Training Loop...")
for epoch in range(epochs):
    for i, data in enumerate(dataloader, 0):

        ############################
        # Discriminator
        ###########################
        # Real Data
        discriminator.zero_grad()
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        output = discriminator(real_cpu).view(-1)
        discriminator_real_loss = criterion(output, label)
        discriminator_real_loss.backward()
        discriminator_real_mean = output.mean().item()

        # Fake Data
        noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
        fake = generator(noise)
        label.fill_(fake_label)
        output = discriminator(fake.detach()).view(-1)
        discriminator_fake_loss = criterion(output, label)
        discriminator_fake_loss.backward()
        discriminator_fake_mean = output.mean().item()
        discriminator_loss = discriminator_real_loss + discriminator_fake_loss
        optimizer_D.step()

        ############################
        # Generator
        ###########################
        generator.zero_grad()
        label.fill_(real_label)
        output = discriminator(fake).view(-1)
        generator_loss = criterion(output, label)
        generator_loss.backward()
        generator_fake_mean = output.mean().item()
        optimizer_G.step()

        if i % 50 == 0:
            print(
                '[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tDiscriminator Mean of Real: %.4f\tDiscriminator Mean of Fake: %.4f / %.4f'
                % (epoch, epochs, i, len(dataloader),
                   discriminator_loss.item(), generator_loss.item(), discriminator_real_mean, discriminator_fake_mean,
                   generator_fake_mean))

        G_losses.append(generator_loss.item())
        D_losses.append(discriminator_loss.item())
        writer.add_scalar("Generator Loss", generator_loss.item(), epoch)
        writer.add_scalar("Discriminator Loss", discriminator_loss.item(), epoch)
        writer.add_scalar('LR', optimizer_G.param_groups[0]['lr'], epoch)
        reduce_on_plateau.step(generator_loss.item())

        if (iters % 500 == 0) or ((epoch == epochs - 1) and (i == len(dataloader) - 1)):
            with torch.no_grad():
                fake = generator(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
            plt.show()

        iters += 1

plt.figure(figsize=(8, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

# %%capture
fig = plt.figure(figsize=(8, 8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
HTML(ani.to_jshtml())

real_batch = next(iter(dataloader))

plt.figure(figsize=(15, 15))
plt.subplot(1, 2, 1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

# Plot the fake images from the last epoch
plt.subplot(1, 2, 2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
plt.show()
