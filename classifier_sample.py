import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import utils

# This is bad! Last-minute implementation now
from params import *

from diffusion.network import MyUNet
from diffusion.DDPM import MyDDPM, generate_new_images

from classifier.network import Conv4Cos

from datasets.all_datasets import get_train_loader

def main():
    # Load data loader
    diffusion_loader, classifier_loader = get_train_loader()
    # Probe data shape
    for batch in diffusion_loader:
        B, C, H, W = batch[0].shape
        break
    # Loading the trained model
    best_model = MyDDPM(MyUNet(), n_steps=n_steps, device=device)
    assert os.path.exists(store_path)

    best_model.load_state_dict(torch.load(store_path, map_location=device))
    best_model.eval()
    print("Diffusion Model loaded")

    # Load classifier
    classifier_weight_path = store_path.replace(".pt", "_classifier.pt")

    classifier_model = Conv4Cos((C, H, W), n_classes=10).to(device)
    classifier_model.load_state_dict(torch.load(classifier_weight_path, map_location=device))
    classifier_model.eval()
    print("Classifier loaded")

    def cond_fn(x, t, y, classifier_scale=0.5):
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier_model(x_in)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return torch.autograd.grad(selected.sum(), x_in)[0] * classifier_scale

    print("Generating new images")
    generated = generate_new_images(
            best_model,
            n_samples=100,
            device=device,
            gif_name="generation_progress_viz.gif",
            cond_fn=cond_fn,
        )
    utils.show_images(generated, "Final result")

if __name__ == "__main__":
    main()
