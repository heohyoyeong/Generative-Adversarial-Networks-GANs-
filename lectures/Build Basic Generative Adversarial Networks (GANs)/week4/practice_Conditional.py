import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from utilsfunction import normalutils, torchutils, torch_models
from tqdm.auto import tqdm
normalutils.set_random_seed(0)


def get_one_hot_labels(labels, n_classes):
    '''
    Function for creating one-hot vectors for the labels, returns a tensor of shape (?, num_classes).
    Parameters:
        labels: tensor of labels from the dataloader, size (?)
        n_classes: the total number of classes in the dataset, an integer scalar
    '''
    result = F.one_hot(labels % n_classes)
    return result


def combine_vectors(x, y):
    # Note: Make sure this function outputs a float no matter what inputs it receives
    x = x.type(torch.float32)
    y = y.type(torch.float32)
    combined = torch.cat((x, y), 1)
    return combined


def get_input_dimensions(z_dim, mnist_shape, n_classes):
    generator_input_dim = z_dim + n_classes
    discriminator_im_chan = mnist_shape[0] + n_classes
    return generator_input_dim, discriminator_im_chan


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    criterion = nn.BCEWithLogitsLoss()
    n_epochs = 200
    z_dim = 64
    display_step = 500
    batch_size = 128
    lr = 0.0002
    device = "cuda"
    save = "./result"
    root = './dataset'

    mnist_shape = (1, 28, 28)
    n_classes = 10

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    dataloader = torchutils.Mnist_dataLoader(root,save,batch_size,transform)

    generator_input_dim, discriminator_im_chan = get_input_dimensions(z_dim, mnist_shape, n_classes)

    gen = torch_models.Basic_Generator(input_dim=generator_input_dim).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
    disc = torch_models.Basic_Discriminator(im_chan=discriminator_im_chan).to(device)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)

    gen = gen.apply(weights_init)
    disc = disc.apply(weights_init)

    # UNQ_C4 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
    # GRADED CELL
    cur_step = 0
    generator_losses = []
    discriminator_losses = []

    # UNIT TEST NOTE: Initializations needed for grading
    noise_and_labels = False
    fake = False

    fake_image_and_labels = False
    real_image_and_labels = False
    disc_fake_pred = False
    disc_real_pred = False

    for epoch in range(n_epochs):
        # Dataloader returns the batches and the labels
        for real, labels in tqdm(dataloader):
            cur_batch_size = len(real)
            # Flatten the batch of real images from the dataset
            real = real.to(device)

            one_hot_labels = get_one_hot_labels(labels.to(device), n_classes)
            image_one_hot_labels = one_hot_labels[:, :, None, None]
            image_one_hot_labels = image_one_hot_labels.repeat(1, 1, mnist_shape[1], mnist_shape[2])

            ### Update discriminator ###
            # Zero out the discriminator gradients
            disc_opt.zero_grad()
            # Get noise corresponding to the current batch_size
            fake_noise = torchutils.get_noise(cur_batch_size, z_dim, device=device)

            # Now you can get the images from the generator
            # Steps: 1) Combine the noise vectors and the one-hot labels for the generator
            #        2) Generate the conditioned fake images

            #### START CODE HERE ####
            noise_and_labels = combine_vectors(fake_noise, one_hot_labels)
            fake = gen(noise_and_labels)
            #### END CODE HERE ####

            # Make sure that enough images were generated
            assert len(fake) == len(real)
            # Check that correct tensors were combined
            assert tuple(noise_and_labels.shape) == (cur_batch_size, fake_noise.shape[1] + one_hot_labels.shape[1])
            # It comes from the correct generator
            assert tuple(fake.shape) == (len(real), 1, 28, 28)

            # Now you can get the predictions from the discriminator
            # Steps: 1) Create the input for the discriminator
            #           a) Combine the fake images with image_one_hot_labels,
            #              remember to detach the generator (.detach()) so you do not backpropagate through it
            #           b) Combine the real images with image_one_hot_labels
            #        2) Get the discriminator's prediction on the fakes as disc_fake_pred
            #        3) Get the discriminator's prediction on the reals as disc_real_pred

            #### START CODE HERE ####
            fake.detach()
            fake_image_and_labels = combine_vectors(fake, image_one_hot_labels)
            real_image_and_labels = combine_vectors(real, image_one_hot_labels)
            disc_fake_pred = disc(fake_image_and_labels)
            disc_real_pred = disc(real_image_and_labels)

            disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
            disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
            disc_loss = (disc_fake_loss + disc_real_loss) / 2
            disc_loss.backward(retain_graph=True)
            disc_opt.step()

            # Keep track of the average discriminator loss
            discriminator_losses += [disc_loss.item()]

            ### Update generator ###
            # Zero out the generator gradients
            gen_opt.zero_grad()

            fake_image_and_labels = combine_vectors(fake, image_one_hot_labels)
            # This will error if you didn't concatenate your labels to your image correctly
            disc_fake_pred = disc(fake_image_and_labels)
            gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
            gen_loss.backward()
            gen_opt.step()

            # Keep track of the generator losses
            generator_losses += [gen_loss.item()]
            #

            if cur_step % display_step == 0 and cur_step > 0:
                gen_mean = sum(generator_losses[-display_step:]) / display_step
                disc_mean = sum(discriminator_losses[-display_step:]) / display_step
                print(f"Epoch {epoch}, step {cur_step}: Generator loss: {gen_mean}, discriminator loss: {disc_mean}")
                torchutils.save_tensor_images(fake, save_loc=save + f"/{epoch}_fake.png")
                torchutils.save_tensor_images(real, save_loc=save + f"/{epoch}_real.png")
                step_bins = 20
                x_axis = sorted([i * step_bins for i in range(len(generator_losses) // step_bins)] * step_bins)
                num_examples = (len(generator_losses) // step_bins) * step_bins
            #     plt.plot(
            #         range(num_examples // step_bins),
            #         torch.Tensor(generator_losses[:num_examples]).view(-1, step_bins).mean(1),
            #         label="Generator Loss"
            #     )
            #     plt.plot(
            #         range(num_examples // step_bins),
            #         torch.Tensor(discriminator_losses[:num_examples]).view(-1, step_bins).mean(1),
            #         label="Discriminator Loss"
            #     )
            #     plt.legend()
            #     plt.show()
            # elif cur_step == 0:
            #     print(
            #         "Congratulations! If you've gotten here, it's working. Please let this train until you're happy with how the generated numbers look, and then go on to the exploration!")
            cur_step += 1