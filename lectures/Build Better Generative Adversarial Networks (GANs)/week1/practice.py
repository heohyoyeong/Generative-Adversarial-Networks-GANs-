import torch
import numpy as np
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utilsfunction import torchutils, torch_models, normalutils
import pandas as pd
from torch.distributions import MultivariateNormal
import seaborn as sns

normalutils.set_random_seed(0)


class new_Generator(torch_models.Basic_Generator):
    def __init__(self, z_dim=10, im_chan=3, hidden_dim=64):
        super(new_Generator, self).__init__()
        self.z_dim = z_dim
        # Build the neural network
        self.gen = nn.Sequential(
            self.make_gen_block(z_dim, hidden_dim * 8),
            self.make_gen_block(hidden_dim * 8, hidden_dim * 4),
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2),
            self.make_gen_block(hidden_dim * 2, hidden_dim),
            self.make_gen_block(hidden_dim, im_chan, kernel_size=4, final_layer=True),
        )

def preprocess(img):
    img = torch.nn.functional.interpolate(img, size=(299, 299), mode='bilinear', align_corners=False)
    return img

def get_covariance(features):
    return torch.Tensor(np.cov(features.detach().numpy(), rowvar=False))

if __name__=="__main__":
    z_dim = 64
    image_size = 299
    device = 'cuda'

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = CelebA(".", download=True, transform=transform)

    gen = new_Generator(z_dim).to(device)
    gen.load_state_dict(torch.load(f"pretrained_celeba.pth", map_location=torch.device(device))["gen"])
    gen = gen.eval()

    from torchvision.models import inception_v3

    inception_model = inception_v3(pretrained=True)
    inception_model.to(device)
    inception_model = inception_model.eval()  # Evaluation mode
    inception_model.fc = nn.Identity(100)

    fake_features_list = []
    real_features_list = []

    gen.eval()
    n_samples = 512  # The total number of samples
    batch_size = 4  # Samples per iteration

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True)

    cur_samples = 0
    with torch.no_grad():  # You don't need to calculate gradients here, so you do this to save memory
        try:
            for real_example, _ in tqdm(dataloader, total=n_samples // batch_size):  # Go by batch
                real_samples = real_example
                real_features = inception_model(real_samples.to(device)).detach().to('cpu')  # Move features to CPU
                real_features_list.append(real_features)

                fake_samples = torchutils.get_noise(len(real_example), z_dim).to(device)
                fake_samples = preprocess(gen(fake_samples))
                fake_features = inception_model(fake_samples.to(device)).detach().to('cpu')
                fake_features_list.append(fake_features)
                cur_samples += len(real_samples)
                if cur_samples > n_samples:
                    break
        except:
            print("Error in loop")

    fake_features_all = torch.cat(fake_features_list)
    real_features_all = torch.cat(real_features_list)

    mu_fake = fake_features_all.mean(0)
    mu_real = real_features_all.mean(0)
    sigma_fake = get_covariance(fake_features_all)
    sigma_real = get_covariance(real_features_all)

    indices = [2, 4, 5]
    fake_dist = MultivariateNormal(mu_fake[indices], sigma_fake[indices][:, indices])
    fake_samples = fake_dist.sample((5000,))
    real_dist = MultivariateNormal(mu_real[indices], sigma_real[indices][:, indices])
    real_samples = real_dist.sample((5000,))


    df_fake = pd.DataFrame(fake_samples.numpy(), columns=indices)
    df_real = pd.DataFrame(real_samples.numpy(), columns=indices)
    df_fake["is_real"] = "no"
    df_real["is_real"] = "yes"
    df = pd.concat([df_fake, df_real])
    sns.pairplot(data=df, plot_kws={'alpha': 0.1}, hue='is_real')
    plt.show()