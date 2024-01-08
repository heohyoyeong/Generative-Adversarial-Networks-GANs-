import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from utilsfunction import normalutils, torchutils
import os
normalutils.set_random_seed(0)


class Generator(nn.Module):
    def __init__(self, z_dim=10, im_chan=1, hidden_dim=64):
        super(Generator,self).__init__()
        self.z_dim = z_dim
        self.gen =nn.Sequential(
            self.make_gen_block(z_dim,hidden_dim*4),
            self.make_gen_block(hidden_dim*4, hidden_dim * 2, kernel_size=4,stride=1),
            self.make_gen_block(hidden_dim*2, hidden_dim),
            self.make_gen_block(hidden_dim, im_chan,kernel_size=4,final_layer=True),
        )

    def make_gen_block(self,input_channel, output_channel, kernel_size=3, stride=2, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channel,output_channel,kernel_size,stride),
                nn.BatchNorm2d(output_channel),
                nn.ReLU()
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channel,output_channel,kernel_size,stride),
                nn.Tanh()
            )

    def unsqueeze_noise(self,noise):
        return noise.view(len(noise), self.z_dim,1,1)

    def forward(self, noise):
        x = self.unsqueeze_noise(noise)
        return self.gen(x)


def get_noise(n_samples, z_dim, device="cpu"):
    return torch.randn(n_samples, z_dim, device=device)


class Discriminator(nn.Module):
    def __init__(self, im_chan=1, hidden_dim=16):
        super(Discriminator,self).__init__()
        self.disc = nn.Sequential(
            self.make_gen_block(im_chan, hidden_dim),
            self.make_gen_block(hidden_dim, hidden_dim*2),
            self.make_gen_block(hidden_dim*2, 1, final_layer=True)
        )

    def make_gen_block(self,input_channel, output_channel, kernel_size=3, stride=2, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channel,output_channel,kernel_size,stride),
                nn.BatchNorm2d(output_channel),
                nn.ReLU()
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channel,output_channel,kernel_size,stride)
            )

    def forward(self,image):
        disc_pred = self.disc(image)
        return disc_pred.view(len(disc_pred),-1)

if __name__ == "__main__":
    criterion = nn.BCEWithLogitsLoss()
    z_dim = 64
    display_step = 500
    batch_size = 128
    # A learning rate of 0.0002 works well on DCGAN
    lr = 0.0002

    # These parameters control the optimizer's momentum, which you can read more about here:
    # https://distill.pub/2017/momentum/ but you don’t need to worry about it for this course!
    beta_1 = 0.5
    beta_2 = 0.999
    device = 'cuda'

    # You can tranform the image values to be between -1 and 1 (the range of the tanh activation)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])


    save = "./result"
    root = './dataset'

    if not os.path.exists(root):
        os.mkdir(root)
        dataloader = DataLoader(
            MNIST(root, download=True, transform=transform),
            batch_size=batch_size,
            shuffle=True)
    else:
        dataloader = DataLoader(
            MNIST(root, download=False, transform=transform),
            batch_size=batch_size,
            shuffle=True)

    if not os.path.exists(save):
        os.mkdir(save)


    gen = Generator(z_dim).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
    disc = Discriminator().to(device)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr, betas=(beta_1, beta_2))


    # You initialize the weights to the normal distribution
    # with mean 0 and standard deviation 0.02
    def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)


    gen = gen.apply(weights_init)
    disc = disc.apply(weights_init)

    n_epochs = 50
    cur_step = 0
    mean_generator_loss = 0
    mean_discriminator_loss = 0
    for epoch in range(n_epochs):
        # Dataloader returns the batches
        for real, _ in tqdm(dataloader):
            cur_batch_size = len(real)
            real = real.to(device)

            ## Update discriminator ##
            disc_opt.zero_grad()
            fake_noise = get_noise(cur_batch_size, z_dim, device=device)
            fake = gen(fake_noise)
            disc_fake_pred = disc(fake.detach())
            disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
            disc_real_pred = disc(real)
            disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
            disc_loss = (disc_fake_loss + disc_real_loss) / 2

            # Keep track of the average discriminator loss
            mean_discriminator_loss += disc_loss.item() / display_step
            # Update gradients
            disc_loss.backward(retain_graph=True)
            # Update optimizer
            disc_opt.step()

            ## Update generator ##
            gen_opt.zero_grad()
            fake_noise_2 = get_noise(cur_batch_size, z_dim, device=device)
            fake_2 = gen(fake_noise_2)
            disc_fake_pred = disc(fake_2)
            gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
            gen_loss.backward()
            gen_opt.step()

            # Keep track of the average generator loss
            mean_generator_loss += gen_loss.item() / display_step

            ## Visualization code ##
            if cur_step % display_step == 0 and cur_step > 0:
                print(
                    f"Epoch {epoch}, step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
                torchutils.save_tensor_images(fake, save_loc=save + f"/{epoch}_fake.png")
                torchutils.save_tensor_images(real, save_loc=save + f"/{epoch}_real.png")
                mean_generator_loss = 0
                mean_discriminator_loss = 0
            cur_step += 1
