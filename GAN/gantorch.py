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


def images_to_vectors(images):
    return images.view(images.size(0), 784)

def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 1, 28, 28)

# discriminator network
class DiscriminatorNet(torch.nn.Module):

    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        
        num_features = 784
        num_outputs = 1

        self.hidden0 = nn.Sequential(
            nn.Linear(num_features, 1024), 
            nn.LeakyReLU(0.2), 
            nn.Dropout(0.3)
        )

        self.hidden1 = nn.Sequential(
            nn.Linear(1024, 512), 
            nn.LeakyReLU(0.2), 
            nn.Dropout(0.3)
        )

        self.hidden2 = nn.Sequential(
            nn.Linear(512, 256), 
            nn.LeakyReLU(0.2), 
            nn.Dropout(0.3)
        )

        self.out = nn.Sequential(
            nn.Linear(256, num_outputs), 
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        
        return self.out(x)

discriminator = DiscriminatorNet()