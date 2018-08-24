import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets

def mnist_data():

    compose = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))]
    )

    out_dir = './dataset'

    return datasets.MNIST(root = out_dir, train = True, transform = compose, download = True)

data = mnist_data()

data_loader = torch.utils.data.DataLoader(data, batch_size = 100, shuffle = True)
num_batches = len(data_loader)