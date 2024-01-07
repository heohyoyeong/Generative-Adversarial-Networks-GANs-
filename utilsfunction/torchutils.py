from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt


def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()

def save_tensor_images(image_tensor, size=(1, 28, 28), save_loc=""):
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image1 = image_unflat[0]
    save_image(image1, save_loc)
