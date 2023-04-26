import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Lambda

# This is bad! Last-minute implementation now
from params import *

# Loading the data (converting each image into a tensor and normalizing between [-1, 1])
transform = Compose([
    ToTensor(),
    Lambda(lambda x: (x - 0.5) * 2)]
)

# Dataset wrapper that filters classes and number of samples per class
class FilteredDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, classes, samples_per_class=99999999999):
        self.dataset = dataset
        self.classes = classes
        self.samples_per_class = samples_per_class

        self.indices = []
        self.cls_cnt = {c: 0 for c in self.classes}
        for i, (x, y) in enumerate(self.dataset):
            if y in self.classes and self.cls_cnt[y] < self.samples_per_class:
                self.indices.append(i)
                self.cls_cnt[y] += 1
        assert len(self.indices) <= self.samples_per_class * len(self.classes)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]

    def __len__(self):
        return len(self.indices)

def get_mnist_diffusion_train_set():
    dataset = torchvision.datasets.MNIST("./datasets", download=True, train=True, transform=transform)
    dataset = FilteredDataset(dataset, classes=[0, 1, 2, 3, 4, 5, 6, 7])
    return dataset

def get_mnist_classifier_train_set():
    dataset = torchvision.datasets.MNIST("./datasets", download=True, train=True, transform=transform)
    dataset = FilteredDataset(dataset, classes=[8, 9], samples_per_class=n_shot)
    return dataset

def get_fashion_diffusion_train_set():
    dataset = torchvision.datasets.FashionMNIST("./datasets", download=True, train=True, transform=transform)
    dataset = FilteredDataset(dataset, classes=[0, 1, 2, 3, 4, 5, 6, 7])
    return dataset

def get_fashion_classifier_train_set():
    dataset = torchvision.datasets.FashionMNIST("./datasets", download=True, train=True, transform=transform)
    dataset = FilteredDataset(dataset, classes=[8, 9], samples_per_class=n_shot)
    return dataset

def get_omniglot_diffusion_train_set():
    raise NotImplementedError

def get_omniglot_classifier_train_set():
    raise NotImplementedError

def get_cifar10_diffusion_train_set():
    raise NotImplementedError

def get_cifar10_classifier_train_set():
    raise NotImplementedError

def get_loader():
    if dataset_name == "mnist":
        diffusion_set = get_mnist_diffusion_train_set()
        classifier_set = get_mnist_classifier_train_set()
    elif dataset_name == "fashion":
        diffusion_set = get_fashion_diffusion_train_set()
        classifier_set = get_fashion_classifier_train_set()
    elif dataset_name == "omniglot":
        diffusion_set = get_omniglot_diffusion_train_set()
        classifier_set = get_omniglot_classifier_train_set()
    elif dataset_name == "cifar10":
        diffusion_set = get_cifar10_diffusion_train_set()
        classifier_set = get_cifar10_classifier_train_set()
    else:
        raise ValueError(f"Dataset {dataset_name} not implemented")
    diffusion_loader = DataLoader(diffusion_set, batch_size, shuffle=True)
    classifier_loader = DataLoader(classifier_set, batch_size, shuffle=True)
    return diffusion_loader, classifier_loader
