import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm import tqdm

import utils

# This is bad! Last-minute implementation now
from params import *

from diffusion.network import MyUNet
from diffusion.DDPM import MyDDPM, generate_new_images

from classifier.network import Conv4Cos

from datasets.all_datasets import get_train_loader

def training_loop(ddpm, loader, n_epochs, optim, device, store_path="ddpm_model.pt"):
    mse = nn.MSELoss()
    best_loss = float("inf")
    n_steps = ddpm.n_steps

    for epoch in tqdm(range(n_epochs), desc=f"Training progress", colour="#00ff00"):
        epoch_loss = 0.0
        for step, batch in enumerate(tqdm(loader, leave=False, desc=f"Epoch {epoch + 1}/{n_epochs}", colour="#005500")):
            # Loading data
            x0 = batch[0].to(device)
            n = len(x0)

            # Picking some noise for each of the images in the batch, a timestep and the respective alpha_bars
            eta = torch.randn_like(x0).to(device)
            t = torch.randint(0, n_steps, (n,)).to(device)

            # Computing the noisy image based on x0 and the time-step (forward process)
            noisy_imgs = ddpm(x0, t, eta)

            # Getting model estimation of noise based on the images and the time-step
            eta_theta = ddpm.backward(noisy_imgs, t.reshape(n, -1))

            # Optimizing the MSE between the noise plugged and the predicted noise
            loss = mse(eta_theta, eta)
            optim.zero_grad()
            loss.backward()
            optim.step()

            epoch_loss += loss.item() * len(x0) / len(loader.dataset)

def main():
    # Loading the trained model
    # Conceptually, the trained model is not necessary for training classifier,
    # since we only need the non-parametric forward diffusion process.
    ddpm_model = MyDDPM(MyUNet(), n_steps=n_steps, device=device)
    ddpm_model.eval()

    diffusion_loader, classifier_loader = get_train_loader()

    # Probe data shape
    for batch in diffusion_loader:
        B, C, H, W = batch[0].shape
        break

    classifier_model = Conv4Cos((C, H, W), n_classes=10).to(device)

    print(classifier_model(batch[0]).shape)

    # First train the classifier end-to-end on the diffusion loader


    # Then, fine-tune the linear layer on the classifier loader

    # Optionally, show the diffusion (forward) process
    # utils.show_forward(ddpm_model, classifier_loader, device)

if __name__ == "__main__":
    main()
