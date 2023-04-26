import torch
import matplotlib.pyplot as plt

import utils

# This is bad! Last-minute implementation now
from params import *

from diffusion.network import MyUNet
from diffusion.DDPM import MyDDPM, generate_new_images

def main():
    # Loading the trained model
    best_model = MyDDPM(MyUNet(), n_steps=n_steps, device=device)
    best_model.load_state_dict(torch.load(store_path, map_location=device))
    best_model.eval()
    print("Model loaded")

    print("Generating new images")
    generated = generate_new_images(
            best_model,
            n_samples=100,
            device=device,
            gif_name="generation_progress_viz.gif"
        )
    # utils.show_images(generated, "Final result")

    # Optionally, show the diffusion (forward) process
    # utils.show_forward(best_model, loader, device)

if __name__ == "__main__":
    main()
