from environment import CatchEnv
import torch as torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.v2 as tf2
from skimage.transform import resize


if __name__ == '__main__':
    env = CatchEnv()
    obs, info = env.reset()
    obs, r, t, _, info = env.step(2)
    obs, r, t, _, info = env.step(0)

    obs = torch.from_numpy(obs).float() / 255.0
    obs = torch.moveaxis(obs, -1, 0)
    rsz = tf2.Resize((21, 21))

    img = rsz(obs)
    img = torch.where(img > 0.25, torch.tensor(1.0), torch.tensor(0.0))
    img = torch.moveaxis(img, 0, -1)
    img = img.numpy()

    fig, ax = plt.subplots(2, 2)
    ax[0][0].imshow(img[:, :, 0], cmap='gray')
    ax[0][1].imshow(img[:, :, 1], cmap='gray')
    ax[1][0].imshow(img[:, :, 2], cmap='gray')
    ax[1][1].imshow(img[:, :, 3], cmap='gray')
    plt.show()





