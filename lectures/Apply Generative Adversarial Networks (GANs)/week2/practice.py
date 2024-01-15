import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch import nn
from skimage import io
from utilsfunction import torchutils, torch_models, normalutils


normalutils.set_random_seed(42)


if __name__ == "__main__":

    criterion = nn.BCEWithLogitsLoss()
    n_epochs = 200
    input_dim = 1
    label_dim = 1
    display_step = 20
    batch_size = 4
    lr = 0.0002
    initial_shape = 512
    target_shape = 373
    device = 'cuda'

    volumes = torch.Tensor(io.imread('train-volume.tif'))[:, None, :, :] / 255
    labels = torch.Tensor(io.imread('train-labels.tif', plugin="tifffile"))[:, None, :, :] / 255
    labels = torchutils.crop(labels, torch.Size([len(labels), 1, target_shape, target_shape]))
    dataset = torch.utils.data.TensorDataset(volumes, labels)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True)
    unet = torch_models.UNet(input_dim, label_dim).to(device)
    unet_opt = torch.optim.Adam(unet.parameters(), lr=lr)
    cur_step = 0

    for epoch in range(n_epochs):
        for real, labels in tqdm(dataloader):
            cur_batch_size = len(real)
            # Flatten the image
            real = real.to(device)
            labels = labels.to(device)

            unet_opt.zero_grad()
            pred = unet(real)
            unet_loss = criterion(pred, labels)
            unet_loss.backward()
            unet_opt.step()

            if cur_step % display_step == 0:
                print(f"Epoch {epoch}: Step {cur_step}: U-Net loss: {unet_loss.item()}")
                torchutils.show_tensor_images(
                    torchutils.crop(real, torch.Size([len(real), 1, target_shape, target_shape])),
                    size=(input_dim, target_shape, target_shape)
                )
                torchutils.show_tensor_images(labels, size=(label_dim, target_shape, target_shape))
                torchutils.show_tensor_images(torch.sigmoid(pred), size=(label_dim, target_shape, target_shape))
            cur_step += 1