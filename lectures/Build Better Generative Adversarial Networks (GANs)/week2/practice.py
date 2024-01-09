import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import imageio

device = "cuda" if torch.cuda.is_available() else "cpu"

def batch_generator(inputs, batch_size):
    """
    Generates batches of `batch_size` from `inputs` array.
    """
    l = inputs.shape[0]
    for i in range(0, l, batch_size):
        yield inputs[i:min(i + batch_size, l)]

# if not os.path.exists('tiny_nerf_data.npz'):
#     wget "https://bmild.github.io/nerf/tiny_nerf_data.npz"

data = np.load('tiny_nerf_data.npz')
images = data['images']
poses = data['poses']
focal = data['focal']
print(images.shape, poses.shape, focal)

testimg, testpose = images[101], poses[101]
# use the first 100 images for training
images = images[:100,...,:3]
poses = poses[:100]

plt.imshow(testimg)
plt.show()

images = torch.from_numpy(images).to(device)
poses = torch.from_numpy(poses).to(device)
testimg = torch.from_numpy(testimg).to(device)
testpose = torch.from_numpy(testpose).to(device)

def get_rays(height, width, focal_length, cam2world):
    """
    Compute the rays (origins and directions) passing through an image with
    `height` and `width` (in pixels). `focal_length` (in pixels) is a property
    of the camera. `cam2world` represents and transform tensor from a 3D point
    in the "camera" frame of reference to the "world" frame of reference (the
    `pose` in our dataset).
    """
    i, j = torch.meshgrid(torch.arange(width).to(cam2world),torch.arange(height).to(cam2world))
    # , indexing = "xy"
    dirs = torch.stack([
        (i.cpu() - width / 2) / focal_length,
        - (j.cpu() - height / 2) / focal_length,
        - torch.ones_like(i.cpu())
    ], dim=-1).to(cam2world)
    rays_d = torch.sum(dirs[..., None, :] * cam2world[:3, :3], dim=-1)
    rays_o = cam2world[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d

def positional_encoding(x, L_embed=6):
    """
    Returns tensor representing positional encoding $\gamma(x)$ of `x` with
    `L_embed` corresponding to $L$ in the above.
    """
    rets = [x]
    for i in range(L_embed):
        for fn in [torch.sin, torch.cos]:
            rets.append(fn(2 ** i * x))
    return torch.cat(rets, dim=-1)

class TinyNeRF(nn.Module):
    """
    Implements 4 layer MLP as a tiny example of the NeRF design
    """
    def __init__(self, hidden_dim=128, L_embed=6):
        super().__init__()
        in_dim = 3 + 3 * 2 * L_embed
        self.layer1 = nn.Linear(in_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim + in_dim, hidden_dim)
        self.layer4 = nn.Linear(hidden_dim, 4)

    def forward(self, x):
        out = F.relu(self.layer1(x))
        out = F.relu(self.layer2(out))
        out = F.relu(self.layer3(torch.cat([out, x], dim=-1)))
        out = self.layer4(out)
        return out

def render_rays(
    model, rays_o, rays_d, near, far, N_samples, encoding_fn, rand=True
):
    """
    Use `model` to render the rays parameterized by `rays_o` and `rays_d`
    between `near` and `far` limits with `N_samples`.
    """
    # sample query pts
    z_vals = torch.linspace(near, far, N_samples).to(rays_o)
    if rand:
        z_vals = (
            torch.rand(list(rays_o.shape[:-1]) + [N_samples])
            * (far - near) / N_samples
        ).to(rays_o) + z_vals
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

    # run query pts through model to get radiance fields
    pts_flat = pts.reshape((-1, 3))
    encoded_pts_flat = encoding_fn(pts_flat)
    batches = batch_generator(encoded_pts_flat, batch_size=BATCH_SIZE)
    preds = []
    for batch in batches:
        preds.append(model(batch))
    radiance_fields_flat = torch.cat(preds, dim=0)
    radiance_fields = torch.reshape(
        radiance_fields_flat, list(pts.shape[:-1]) + [4]
    )

    # compute densities and colors
    sigma_a = F.relu(radiance_fields[..., 3])
    rgb = torch.sigmoid(radiance_fields[..., :3])

    # do volume rendering
    oneE10 = torch.tensor([1e10], dtype=rays_o.dtype, device=rays_o.device)
    dists = torch.cat([
        z_vals[..., 1:] - z_vals[..., :-1],
        oneE10.expand(z_vals[..., :1].shape)
    ], dim=-1)
    alpha = 1 - torch.exp(-sigma_a * dists)
    weights = torch.roll(torch.cumprod(1 - alpha + 1e-10, dim=-1), 1, dims=-1)
    weights[..., 0] = 1
    weights = alpha * weights

    rgb_map = (weights[..., None] * rgb).sum(dim=-2)
    depth_map = (weights * z_vals).sum(dim=-1)
    acc_map = weights.sum(dim=-1)
    return rgb_map, depth_map, acc_map


def pose_spherical(theta, phi, radius):
    """
    Compute a transformation tensor for a spherical coordinates
    (`theta`, `phi`, `radius`)
    """
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w.numpy()
    return c2w


if __name__=="__main__":
    # define parameters
    NUM_ENCODING_FUNCTIONS = 6
    NEAR = 2
    FAR = 6
    DEPTH_SAMPLES = 64
    LEARNING_RATE = 4e-3
    BATCH_SIZE = 16384
    NUM_EPOCHS = 1000
    DISPLAY_EVERY = 100
    HEIGHT, WIDTH = images.shape[1:3]
    FOCAL = data['focal']
    # SEED = 42
    # torch.manual_seed(SEED)
    # np.random.seed(SEED)

    # initialize encoding function, model, loss, and optimizer
    encoding_fn = lambda x: positional_encoding(x, L_embed=NUM_ENCODING_FUNCTIONS)
    model = TinyNeRF(L_embed=NUM_ENCODING_FUNCTIONS)
    model.to(device)
    loss_fn = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # for plotting the loss and iteration during training
    psnrs = []
    iternums = []

    for i in range(NUM_EPOCHS + 1):
        # sample an image from our training set
        img_idx = np.random.randint(images.shape[0])
        target = images[img_idx].to(device)
        pose = poses[img_idx].to(device)

        # get the rays passing through the image and forward pass the model
        rays_o, rays_d = get_rays(HEIGHT, WIDTH, FOCAL, pose)
        rgb, _, _ = render_rays(
            model, rays_o, rays_d, near=NEAR, far=FAR, N_samples=DEPTH_SAMPLES,
            encoding_fn=encoding_fn
        )

        # backward pass
        loss = loss_fn(rgb, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # plot the model's render of the test image and loss at each iteration
        if i % DISPLAY_EVERY == 0:
            rays_o, rays_d = get_rays(HEIGHT, WIDTH, FOCAL, testpose)
            rgb, _, _ = render_rays(
                model, rays_o, rays_d, near=NEAR, far=FAR, N_samples=DEPTH_SAMPLES,
                encoding_fn=encoding_fn
            )
            loss = loss_fn(rgb, testimg)
            print(f"Loss: {loss.item()}")
            psnr = -10 * torch.log10(loss)
            psnrs.append(psnr.item())
            iternums.append(i)


    trans_t = lambda t : torch.tensor([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,t],
        [0,0,0,1],
    ], dtype=torch.float32)

    rot_phi = lambda phi : torch.tensor([
        [1,0,0,0],
        [0,np.cos(phi),-np.sin(phi),0],
        [0,np.sin(phi), np.cos(phi),0],
        [0,0,0,1],
    ], dtype=torch.float32)

    rot_theta = lambda th : torch.tensor([
        [np.cos(th),0,-np.sin(th),0],
        [0,1,0,0],
        [np.sin(th),0, np.cos(th),0],
        [0,0,0,1],
    ], dtype=torch.float32)




    # run poses that encircle the object through our trained model and make a video
    frames = []
    for th in np.linspace(0., 360., 120, endpoint=False):
        c2w = pose_spherical(th, -30, 4)
        c2w = torch.from_numpy(c2w).to(device).float()
        rays_o, rays_d = get_rays(HEIGHT, WIDTH, FOCAL, c2w[:3,:4])
        rgb, _, _ = render_rays(
            model, rays_o, rays_d, NEAR, FAR, N_samples=DEPTH_SAMPLES,
            encoding_fn=encoding_fn
        )
        frames.append((255*np.clip(rgb.cpu().detach().numpy(),0,1)).astype(np.uint8))


    f = 'video.mp4'
    imageio.mimwrite(f, frames, fps=30, quality=7)

