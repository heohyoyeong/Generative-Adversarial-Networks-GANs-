import torch
from torch import nn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from utilsfunction import normalutils, torch_models, torchutils
from torchvision.datasets import CIFAR100
from torchvision import transforms

normalutils.set_random_seed(42)


class new_conditional_generator(torch_models.Basic_Generator):
    def __init__(self, input_dim=10, im_chan=3, hidden_dim=64):
        super(new_conditional_generator, self).__init__()
        self.input_dim = input_dim
        # Build the neural network
        self.gen = nn.Sequential(
            self.make_gen_block(input_dim, hidden_dim * 4, kernel_size=4),
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1),
            self.make_gen_block(hidden_dim * 2, hidden_dim, kernel_size=4),
            self.make_gen_block(hidden_dim, im_chan, kernel_size=2, final_layer=True),
        )


class new_conditional_Classifier(torch_models.Basic_Discriminator):
    def __init__(self, im_chan, n_classes, hidden_dim=32):
        super(new_conditional_Classifier, self).__init__()
        self.disc = nn.Sequential(
            self.make_disc_block(im_chan, hidden_dim, kernel_size=3),
            self.make_disc_block(hidden_dim, hidden_dim * 2, kernel_size=3),
            self.make_disc_block(hidden_dim * 2, hidden_dim * 4, kernel_size=3),
            self.make_disc_block(hidden_dim * 4, n_classes, kernel_size=3, final_layer=True),
        )


class new_conditional_Discriminator(torch_models.Basic_Discriminator):
    def __init__(self, im_chan=3, hidden_dim=64):
        super(new_conditional_Discriminator, self).__init__()
        self.disc = nn.Sequential(
            self.make_disc_block(im_chan, hidden_dim, stride=1),
            self.make_disc_block(hidden_dim, hidden_dim * 2),
            self.make_disc_block(hidden_dim * 2, hidden_dim * 4),
            self.make_disc_block(hidden_dim * 4, 1, final_layer=True),
        )


def train_generator():
    cifar100_shape = (3, 32, 32)
    n_classes = 100
    n_epochs = 100
    z_dim = 64
    display_step = 500
    batch_size = 64
    lr = 0.0002
    device = 'cuda'
    generator_input_dim = z_dim + n_classes
    gen = new_conditional_generator(generator_input_dim).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
    discriminator_input_dim = cifar100_shape[0] + n_classes
    disc = new_conditional_Discriminator(discriminator_input_dim).to(device)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)

    def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)

    gen = gen.apply(weights_init)
    disc = disc.apply(weights_init)

    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5,), (0.5,)),
    ])


    dataloader = DataLoader(
        CIFAR100(".", train=False, download=True, transform=transform),
        batch_size=batch_size)


    criterion = nn.BCEWithLogitsLoss()

    cur_step = 0
    mean_generator_loss = 0
    mean_discriminator_loss = 0
    for epoch in range(n_epochs):
        # Dataloader returns the batches and the labels
        for real, labels in dataloader:
            cur_batch_size = len(real)
            # Flatten the batch of real images from the dataset
            real = real.to(device)

            # Convert the labels from the dataloader into one-hot versions of those labels
            one_hot_labels = torchutils.get_one_hot_labels(labels.to(device), n_classes).float()

            image_one_hot_labels = one_hot_labels[:, :, None, None]
            image_one_hot_labels = image_one_hot_labels.repeat(1, 1, cifar100_shape[1], cifar100_shape[2])

            ### Update discriminator ###
            # Zero out the discriminator gradients
            disc_opt.zero_grad()
            # Get noise corresponding to the current batch_size
            fake_noise = torchutils.get_noise(cur_batch_size, z_dim, device=device)

            # Combine the vectors of the noise and the one-hot labels for the generator
            noise_and_labels = torchutils.combine_vectors(fake_noise, one_hot_labels)
            fake = gen(noise_and_labels)
            # Combine the vectors of the images and the one-hot labels for the discriminator
            fake_image_and_labels = torchutils.combine_vectors(fake.detach(), image_one_hot_labels)
            real_image_and_labels = torchutils.combine_vectors(real, image_one_hot_labels)
            disc_fake_pred = disc(fake_image_and_labels)
            disc_real_pred = disc(real_image_and_labels)

            disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
            disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
            disc_loss = (disc_fake_loss + disc_real_loss) / 2
            disc_loss.backward(retain_graph=True)
            disc_opt.step()

            # Keep track of the average discriminator loss
            mean_discriminator_loss += disc_loss.item() / display_step

            ### Update generator ###
            # Zero out the generator gradients
            gen_opt.zero_grad()

            # Pass the discriminator the combination of the fake images and the one-hot labels
            fake_image_and_labels = torchutils.combine_vectors(fake, image_one_hot_labels)

            disc_fake_pred = disc(fake_image_and_labels)
            gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
            gen_loss.backward()
            gen_opt.step()

            # Keep track of the average generator loss
            mean_generator_loss += gen_loss.item() / display_step

            if cur_step % display_step == 0 and cur_step > 0:
                print(
                    f"Step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
                # torchutils.show_tensor_images(fake)
                # torchutils.show_tensor_images(real)
                mean_generator_loss = 0
                mean_discriminator_loss = 0
            cur_step += 1
    torch.save(gen.state_dict(), "gen.pt")
    torch.save(disc.state_dict(), "disc.pt")

def train_classifier():
    criterion = nn.CrossEntropyLoss()
    n_epochs = 100

    display_step = 100
    batch_size = 512
    lr = 0.0002

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    dataloader = DataLoader(
        CIFAR100(".", train=True, download=True, transform=transform),
        batch_size=batch_size)

    validation_dataloader = DataLoader(
        CIFAR100(".", train=False, download=True, transform=transform),
        batch_size=batch_size)


    classifier = new_conditional_Classifier(cifar100_shape[0], n_classes).to(device)

    # classifier.load_state_dict(torch.load("classifier.pt"))

    classifier_opt = torch.optim.Adam(classifier.parameters(), lr=lr)
    cur_step = 0
    for epoch in range(n_epochs):
        for real, labels in tqdm(dataloader):
            cur_batch_size = len(real)
            real = real.to(device)
            labels = labels.to(device)

            ### Update classifier ###
            # Get noise corresponding to the current batch_size
            classifier_opt.zero_grad()
            labels_hat = classifier(real.detach())
            classifier_loss = criterion(labels_hat, labels)
            classifier_loss.backward()
            classifier_opt.step()

            if cur_step % display_step == 0:
                classifier_val_loss = 0
                classifier_correct = 0
                num_validation = 0
                for val_example, val_label in validation_dataloader:
                    cur_batch_size = len(val_example)
                    num_validation += cur_batch_size
                    val_example = val_example.to(device)
                    val_label = val_label.to(device)
                    labels_hat = classifier(val_example)
                    classifier_val_loss += criterion(labels_hat, val_label) * cur_batch_size
                    classifier_correct += (labels_hat.argmax(1) == val_label).float().sum()

                print(f"Step {cur_step}: "
                      f"Classifier loss: {classifier_val_loss.item() / num_validation}, "
                      f"classifier accuracy: {classifier_correct.item() / num_validation}")
            cur_step += 1
    torch.save(classifier.state_dict(),"classifier.pt")

if __name__=="__main__":
    cifar100_shape = (3, 32, 32)
    n_classes = 100
    n_epochs = 100
    z_dim = 64
    display_step = 500
    batch_size = 64
    lr = 0.0002
    device = 'cuda'
    generator_input_dim = z_dim + n_classes
    # train_classifier()
    # train_generator()