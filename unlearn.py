import torch
import torchvision.datasets as datasets
import torch.nn as nn
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


def compute_saliency_map(model, inputs, target_class):
    model.eval()
    inputs.requires_grad = True

    # Zero the gradients of the weights
    model.zero_grad()

    outputs = model(inputs)
    target_class_tensor = torch.tensor([target_class])  # Convert the target class to a tensor
    outputs = outputs.gather(1, target_class_tensor.view(-1, 1)).squeeze()  # Get the outputs for the target class
    outputs.backward(torch.ones_like(outputs))  # Compute the gradients

    # Get the gradients of the weights of the first layer
    weight_gradients = model.fc1.weight.grad.squeeze()
    statsWeights= weight_gradients
    statsWeights[statsWeights < 0] = 0
    print(statsWeights.shape)
    print(np.nanmean(statsWeights), np.nanmax(statsWeights))
    print(np.count_nonzero(statsWeights))
    # Visualize the gradients of the weights
    plt.imshow(weight_gradients.detach().numpy(), cmap=plt.cm.grey)
    plt.axis('off')
    plt.show()

    return weight_gradients

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(12*12*64, 128)
        self.fc2 = nn.Linear(128, 9)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        out = self.fc2(x)
        return out

# Load the model
model = CNN()
model.load_state_dict(torch.load("modelNo9.pth"))



test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

test_mask = test_data.targets == 3
test_data.data = test_data.data[test_mask]
test_data.targets = test_data.targets[test_mask]

#dataloader
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1)

compute_saliency_map(model, torch.tensor(test_data[0][0].unsqueeze(0)), torch.tensor(3))