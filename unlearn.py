import torch
import torchvision.datasets as datasets
import torch.nn as nn
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def compute_all_gradients(model,inputs, target_class):
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

def show_gradient_map(weight_gradients):
    # Visualize the gradients of the weights
    plt.imshow(weight_gradients.detach().numpy(), cmap=plt.cm.gray)
    plt.axis('off')
    plt.show()

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

target=6
test_mask = test_data.targets == target
test_data.data = test_data.data[test_mask]
test_data.targets = test_data.targets[test_mask]

#create the gradient holders
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1)
grads_conv1 = torch.zeros(model.conv1.weight.shape).squeeze()
grads_conv2 = torch.zeros(model.conv2.weight.shape).squeeze()
grads_fc1 = torch.zeros(model.fc1.weight.shape).squeeze()
grads_fc2 = torch.zeros(model.fc2.weight.shape).squeeze()

# Compute the gradients of the weights of all layers for the target class
for img,_ in test_data:
    img = img.unsqueeze(0)
    grads_conv1=compute_gradients(model, model.conv1, img, target).abs()
    grads_conv2 += compute_gradients(model, model.conv2, img, target).abs()
    grads_fc1 += compute_gradients(model, model.fc1, img, target).abs()
    grads_fc2 += compute_gradients(model, model.fc2, img, target).abs()

#test= compute_gradients(model,model.fc2,torch.tensor(test_data[0][0].unsqueeze(0)), torch.tensor(6)).abs()
#test.abs()
print(grads_conv1.shape)
#show_gradient_map(grads_fc1)
print(grads_fc1.shape)
test= grads_fc1.clone().detach()

#take the value of the 10% highest gradients
treshold= torch.topk(test.flatten(), 2000)[0][-1]
#print(treshold[0],test.mean())

test[test < treshold] = 0
print(test.shape)
show_gradient_map(grads_fc1-test)



#print(test_data[0].shape)