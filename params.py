import torch

fashion = True
batch_size = 128
n_epochs = 20
lr = 0.001
n_steps, min_beta, max_beta = 1000, 10 ** -4, 0.02  # Originally used by the authors

store_path = "ddpm_fashion.pt" if fashion else "ddpm_mnist.pt"

# Getting device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\t" + (f"{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"))
