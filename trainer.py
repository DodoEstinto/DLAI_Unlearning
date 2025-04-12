import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


#dowload cminst10 dataset
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)


#download test dataset
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)


target = 0

# Filter the dataset to exclude the target
train_mask = training_data.targets != target 

training_data.data = training_data.data[train_mask]
training_data.targets = training_data.targets[train_mask]

test_mask = test_data.targets != target
test_data.data = test_data.data[test_mask]
test_data.targets = test_data.targets[test_mask]
print(training_data.classes)
print(test_data.classes)
device= 'cuda' if torch.cuda.is_available() else 'cpu'

#create a cnn with2 hidden layers
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, 1)
        torch.nn.init.xavier_normal_(self.conv1.weight)
        self.conv2 = nn.Conv2d(4, 4, 3, 1)
        torch.nn.init.xavier_normal_(self.conv2.weight)
        self.fc1 = nn.Linear(12*12*4, 32)
        torch.nn.init.xavier_normal_(self.fc1.weight)
        self.fc2 = nn.Linear(32, 10)
        torch.nn.init.xavier_normal_(self.fc2.weight)

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

    
def train(dataloader, model, loss_fn, optimizer,scheduler):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    scheduler.step()    


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

#hyperparameters
learning_rate = 5e-2
batch_size = 32
epochs = 4


model = CNN().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

#initialize dataloaders
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer,scheduler)
    test(test_dataloader, model)
print("Done!")

torch.save(model.state_dict(), "models/modelNo"+str(target)+".pth")


