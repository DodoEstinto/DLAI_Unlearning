import torch
import torchvision.datasets as datasets
import torch.nn as nn
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader


'''
Compute and return the gradients of the weights of a specific layer for a specific class and input image
'''
def compute_gradients(model,layer,inputs, target_class):
    model.eval()
    inputs.requires_grad = True

    # Zero the gradients of the weights
    model.zero_grad()

    outputs = model(inputs)
    target_class_tensor = torch.tensor([target_class])  # Convert the target class to a tensor
    #print(outputs)
    outputs = outputs.gather(1, target_class_tensor.view(-1, 1)).squeeze()  # Get the output of the target class
    #print(outputs)
    outputs.backward()  # Compute the gradients

    # Get the gradients of the weights of the first layer
    weight_gradients = layer.weight.grad.squeeze()
    #statsWeights= weight_gradients
    #statsWeights[statsWeights < 0] = 0
    #print(statsWeights.shape)
    #print(np.nanmean(statsWeights), np.nanmax(statsWeights))
    #print(np.count_nonzero(statsWeights))


    return weight_gradients
'''
Visualize the gradients of the weights
'''
def show_gradient_map(weight_gradients):
    plt.imshow(weight_gradients.detach().numpy(), cmap=plt.cm.gray)
    plt.axis('off')
    plt.show()

def calculate_map_and_treshold(weight_gradients,k=2000):
    grad_map= weight_gradients.clone().detach()

    #take the value of the 10% highest gradients
    treshold= torch.topk(grad_map.flatten(),k)[0][-1]

    grad_map[grad_map < treshold] = 0

    return grad_map, treshold

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

# Load the model
model = CNN()
model.load_state_dict(torch.load("modelNo9.pth"))



test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

FORGET_TARGET=6
test_mask = test_data.targets == FORGET_TARGET
test_data.data = test_data.data[test_mask]
test_data.targets = test_data.targets[test_mask]

#create the gradient holders
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1)
#grads_conv1 = torch.zeros(model.conv1.weight.shape).squeeze()
#grads_conv2 = torch.zeros(model.conv2.weight.shape).squeeze()
grads_fc1 = torch.zeros(model.fc1.weight.shape).squeeze()
grads_fc2 = torch.zeros(model.fc2.weight.shape).squeeze()

# Compute the gradients of the weights of all layers for the target class
for img,_ in test_data:
    img = img.unsqueeze(0)
    #grads_conv1=compute_gradients(model, model.conv1, img, target).abs()
    #grads_conv2 += compute_gradients(model, model.conv2, img, target).abs()
    grads_fc1 += compute_gradients(model, model.fc1, img, FORGET_TARGET).abs()
    grads_fc2 += compute_gradients(model, model.fc2, img, FORGET_TARGET).abs()


#takes about 10% of the highest gradients
fc1_map,_=calculate_map_and_treshold(grads_fc1,2000)
fc2_map,_=calculate_map_and_treshold(grads_fc2,28)

################################# Retraining part #################################

SUB_TARGET=9
device=  'cpu'

#dowload cminst10 dataset
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

# Filter the dataset to exclude class 6
train_mask = training_data.targets != FORGET_TARGET
training_data.data = training_data.data[train_mask]
training_data.targets = training_data.targets[train_mask]
training_data.targets[training_data.targets == SUB_TARGET] = FORGET_TARGET


test_mask = test_data.targets != FORGET_TARGET
test_data.data = test_data.data[test_mask]
test_data.targets = test_data.targets[test_mask]
test_data.targets[test_data.targets == SUB_TARGET] = FORGET_TARGET

print(training_data.targets.unique())
print(test_data.targets.unique(),test_data.data.shape)

#train
def train(dataloader, model, loss_fn, optimizer,scheduler):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        #remove the gradients from fc1 and fc2 using the mask
        model.fc1.weight.grad[fc1_map == 0] = 0
        model.fc2.weight.grad[fc2_map == 0] = 0
        optimizer.step()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    scheduler.step()    


#test
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


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

#initialize dataloaders
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

#train and test
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer,scheduler)
    test(test_dataloader, model)
print("Done!")

#save the model
torch.save(model.state_dict(), "modelRetr.pth")



#print(test_data[0].shape)