
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision import datasets

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 4, 3, 1)
        self.conv2 = nn.Conv2d(4, 4, 3, 1)
        self.fc1 = nn.Linear(12*12*4, 32)
        self.fc2 = nn.Linear(32, 9)

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
device='cpu'
learning_rate = 5e-2
batch_size = 32
epochs = 10
loss_fn = nn.CrossEntropyLoss()
def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


SUB_TARGET=9
FORGET_TARGET=6
batch_size = 32
test_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

#model = CNN()
#model.load_state_dict(torch.load("modelRetr.pth"))
test_static = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
test_mask = (test_static.targets != SUB_TARGET) & (test_static.targets != FORGET_TARGET)

test_static.data = test_static.data[test_mask]
test_static.targets = test_static.targets[test_mask]
test_static_dataloader = DataLoader(test_static, batch_size=batch_size)

test_to_learn = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
test_mask = test_to_learn.targets != FORGET_TARGET
test_to_learn.data = test_to_learn.data[test_mask]
test_to_learn.targets = test_to_learn.targets[test_mask]
test_to_learn_dataloader = DataLoader(test_to_learn, batch_size=batch_size)


#test(test_dataloader, model)

TrueMNIST= datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

for i in range(50):
    print(test_to_learn.targets[i],test_static.targets[i])