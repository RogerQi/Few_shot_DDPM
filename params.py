import torch

batch_size = 128
n_epochs = 20
lr = 0.001
n_steps, min_beta, max_beta = 1000, 10 ** -4, 0.02  # Originally used by the authors

dataset_name = 'fashion'
assert dataset_name in ['mnist', 'fashion', 'omniglot', 'cifar10']

store_path = f"ddpm_{dataset_name}.pt"

n_shot = 5

# Getting device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\t" + (f"{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"))
