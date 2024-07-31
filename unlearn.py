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


device=  'cpu'

#targets
FORGET_TARGET=6
SUB_TARGET=9

#hyperparameters
learning_rate = 5e-2
batch_size = 32
epochs = 2

################################# Dataset preprocessing part #################################

# Load and preprocess the datasets.

#this will contain only the data about the new class
test_only_to_learn = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)
test_mask = test_only_to_learn.targets == SUB_TARGET
test_only_to_learn.data = test_only_to_learn.data[test_mask]
test_only_to_learn.targets = test_only_to_learn.targets[test_mask]
test_only_to_learn.targets[test_only_to_learn.targets == SUB_TARGET] = FORGET_TARGET
test__only_to_learn_dataloader = DataLoader(test_only_to_learn, batch_size=batch_size)

#this will contain only the data about the forgotten class
test_only_forgotten_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)
test_mask = test_only_forgotten_data.targets == FORGET_TARGET
test_only_forgotten_data.data = test_only_forgotten_data.data[test_mask]
test_only_forgotten_data.targets = test_only_forgotten_data.targets[test_mask]
test_only_forgotten_dataloader = DataLoader(test_only_forgotten_data, batch_size=batch_size)



#this will contain the training data where the forgotten class is substituted with the new class
training_to_learn = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)
train_mask = training_to_learn.targets != FORGET_TARGET
training_to_learn.data = training_to_learn.data[train_mask]
training_to_learn.targets = training_to_learn.targets[train_mask]
training_to_learn.targets[training_to_learn.targets == SUB_TARGET] = FORGET_TARGET


#this will contain the test data where the forgotten class is substituted with the new class
test_to_learn = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)
test_mask = test_to_learn.targets != FORGET_TARGET
test_to_learn.data = test_to_learn.data[test_mask]
test_to_learn.targets = test_to_learn.targets[test_mask]
test_to_learn.targets[test_to_learn.targets == SUB_TARGET] = FORGET_TARGET


################################# Gradient computation part #################################

# Load the model
model = CNN()
model.load_state_dict(torch.load("modelNo9.pth"))


#create the gradient holders
#grads_conv1 = torch.zeros(model.conv1.weight.shape).squeeze()
#grads_conv2 = torch.zeros(model.conv2.weight.shape).squeeze()
grads_fc1 = torch.zeros(model.fc1.weight.shape).squeeze()
grads_fc2 = torch.zeros(model.fc2.weight.shape).squeeze()

# Compute the gradients of the weights of all layers for the target class
for img,_ in test_only_forgotten_data:
    img = img.unsqueeze(0)
    #grads_conv1=compute_gradients(model, model.conv1, img, target).abs()
    #grads_conv2 += compute_gradients(model, model.conv2, img, target).abs()
    grads_fc1 += compute_gradients(model, model.fc1, img, FORGET_TARGET).abs()
    grads_fc2 += compute_gradients(model, model.fc2, img, FORGET_TARGET).abs()


#takes about 10% of the highest gradients
fc1_map,_=calculate_map_and_treshold(grads_fc1,2000)
fc2_map,_=calculate_map_and_treshold(grads_fc2,28)


################################# Retraining part #################################




print(training_to_learn.targets.unique())
print(test_to_learn.targets.unique(),test_to_learn.data.shape)

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
        if batch % 400 == 0:
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
    print(f"Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")





loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

#save the starting errors
starting_error_forgotten=test(test_only_forgotten_dataloader, model)
starting_error_new=test(test__only_to_learn_dataloader, model)

#initialize dataloaders
train_to_learn_dataloader = DataLoader(training_to_learn, batch_size=batch_size)
test_to_learn_dataloader = DataLoader(test_to_learn, batch_size=batch_size)

#train and test
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_to_learn_dataloader, model, loss_fn, optimizer,scheduler)
    print("Error on dataset:")
    test(test_to_learn_dataloader, model)
    print("Error on forgotten data:")
    test(test_only_forgotten_dataloader, model)
    print("Error on the new data:")
    test(test__only_to_learn_dataloader, model)

print("Final error:")
test(test_to_learn_dataloader, model)
print("Final error on forgotten data:")
test(test_only_forgotten_dataloader, model)
print("Final error on the new data:")
test(test__only_to_learn_dataloader, model)
print("Starting error on forgotten data:\n",starting_error_forgotten)
print("Starting error on the new data:\n",starting_error_new)


#save the model
torch.save(model.state_dict(), "modelRetr.pth")



#print(test_data[0].shape)