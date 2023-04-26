import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.optim import Adam

from tqdm import tqdm

import utils

# This is bad! Last-minute implementation now
from params import *

from diffusion.network import MyUNet
from diffusion.DDPM import MyDDPM, generate_new_images

from classifier.network import Conv4Cos

from datasets.all_datasets import get_train_loader

def training_loop(ddpm, classifier_model, loader, n_epochs, optim, device, store_path):
    criterion = nn.CrossEntropyLoss()
    n_steps = ddpm.n_steps

    total_train_iter_cnt = 0

    for epoch in tqdm(range(n_epochs), desc=f"Training progress", colour="#00ff00"):
        for _, batch in enumerate(loader):
            # Loading data
            x0 = batch[0].to(device)
            y0 = batch[1].to(device)

            n = len(x0)

            # Picking some noise for each of the images in the batch, a timestep and the respective alpha_bars
            eta = torch.randn_like(x0).to(device)
            if False:
                t = torch.randint(0, 1, (n,)).to(device)
            else:
                t = torch.randint(0, n_steps, (n,)).to(device)

            # Computing the noisy image based on x0 and the time-step (forward process)
            noisy_imgs = ddpm(x0, t, eta)

            output = classifier_model(noisy_imgs)

            # Optimizing the MSE between the noise plugged and the predicted noise
            loss = criterion(output, y0)
            optim.zero_grad()
            loss.backward()
            optim.step()

            total_train_iter_cnt += 1

            if total_train_iter_cnt % 100 == 0:
                _, predicted = torch.max(output, 1)
                acc = (predicted == y0).sum().item()
                print(f"Training accuracy at Epoch {epoch} iteration {total_train_iter_cnt}: {acc / n:.3f}")
                print(f"Training loss at iteration {total_train_iter_cnt}: {loss.item():.3f}")

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

    # First train the classifier end-to-end on the diffusion loader
    training_loop(ddpm_model, classifier_model, diffusion_loader, n_epochs, Adam(classifier_model.parameters(), lr=1e-3), device, "classifier_model.pt")

    # Then, fine-tune the linear layer on the classifier loader

    # Optionally, show the diffusion (forward) process
    # utils.show_forward(ddpm_model, classifier_loader, device)

if __name__ == "__main__":
    main()
