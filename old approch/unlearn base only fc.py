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
        self.fc2 = nn.Linear(32, 10)

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


device= 'cuda' if torch.cuda.is_available() else 'cpu'

#targets
FORGET_TARGET=6
SUB_TARGET=9

#hyperparameters
learning_rate = 5e-3


batch_size = 16
epochs_forget = 2
epochs_relearn = 10

################################# Dataset preprocessing part #################################

# Load and preprocess the datasets.

'''
#this will contain only the train data about the new class
train_only_to_learn = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

train_mask = train_only_to_learn.targets == SUB_TARGET
train_only_to_learn.data = train_only_to_learn.data[train_mask]
train_only_to_learn.targets = train_only_to_learn.targets[train_mask]
train_only_to_learn.targets[train_only_to_learn.targets == SUB_TARGET] = FORGET_TARGET
train_only_to_learn_dataloader = DataLoader(train_only_to_learn, batch_size=batch_size)
'''

#this will contain only the train data about the forgotten class
train_only_forgotten_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)
train_mask = train_only_forgotten_data.targets == FORGET_TARGET
train_only_forgotten_data.data = train_only_forgotten_data.data[train_mask]
train_only_forgotten_data.targets = train_only_forgotten_data.targets[train_mask]
train_only_forgotten_dataloader = DataLoader(train_only_forgotten_data, batch_size=batch_size)


#this will contain only the test data about the new class
test_only_to_learn = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

test_mask = test_only_to_learn.targets == SUB_TARGET
test_only_to_learn.data = test_only_to_learn.data[test_mask]
test_only_to_learn.targets = test_only_to_learn.targets[test_mask]
test_only_to_learn_dataloader = DataLoader(test_only_to_learn, batch_size=batch_size)

#this will contain only the data about the forgotten class
test_only_forgotten_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
test_mask = test_only_forgotten_data.targets == FORGET_TARGET
test_only_forgotten_data.data = test_only_forgotten_data.data[test_mask]
test_only_forgotten_data.targets = test_only_forgotten_data.targets[test_mask]
test_only_forgotten_dataloader = DataLoader(test_only_forgotten_data, batch_size=batch_size)



#this will contain the training data where the forgotten class is removed
training_to_learn = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)
train_mask = training_to_learn.targets != FORGET_TARGET
training_to_learn.data = training_to_learn.data[train_mask]
training_to_learn.targets = training_to_learn.targets[train_mask]
training_to_learn_dataloader = DataLoader(training_to_learn, batch_size=batch_size)


#this will contain the test data with the forget class removed
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


'''
training_two_target_learn = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)
train_mask = training_two_target_learn.targets == (FORGET_TARGET or SUB_TARGET)
training_two_target_learn.data = training_two_target_learn.data[train_mask]
training_two_target_learn.targets = training_two_target_learn.targets[train_mask]
training_two_target_learn_dataloader = DataLoader(test_only_to_learn, batch_size=batch_size)
'''
################################# Gradient computation part #################################

def log_softmax(x):
    return x - torch.logsumexp(x,dim=1, keepdim=True)

def CustomCrossEntropyLoss(outputs, targets):
    epsilon=1e-6
    num_examples = targets.shape[0]
    batch_size = outputs.shape[0]
    outputs[targets==FORGET_TARGET]=-outputs[targets==FORGET_TARGET]
    outputs = log_softmax(outputs+epsilon)
    
    #take only the target loss
    outputs = outputs[range(batch_size), targets]

    return - torch.sum(outputs)/num_examples



# Load the model
model = CNN()
model.load_state_dict(torch.load("modelNo9.pth",map_location=torch.device(device)))


#create the gradient holders
grads_conv1 = torch.zeros(model.conv1.weight.shape)
grads_conv2 = torch.zeros(model.conv2.weight.shape)
grads_fc1 = torch.zeros(model.fc1.weight.shape).squeeze()

print(model.conv1.weight.shape)
print(model.conv2.weight.shape)
# Compute the gradients of the weights of all layers for the target class
for img,_ in test_only_forgotten_data:
    img = img.unsqueeze(0)
    grads_conv1=compute_gradients(model, model.conv1, img, FORGET_TARGET).abs()
    grads_conv2 += compute_gradients(model, model.conv2, img, FORGET_TARGET).abs()
    grads_fc1 += compute_gradients(model, model.fc1, img, FORGET_TARGET).abs()


#takes about 10% of the highest gradients
conv1_map,_=calculate_map_and_treshold(grads_conv1,4)
conv1_map=conv1_map.unsqueeze(1)
conv2_map,_=calculate_map_and_treshold(grads_conv2,16)
fc1_map,_=calculate_map_and_treshold(grads_fc1,1000)


################################# Retraining part #################################
model=model.to(device)
#for param in model.conv1.parameters():
#    param.requires_grad = False

#for param in model.conv2.parameters():
#    param.requires_grad = False


# Define a custom backward hook to zero out gradients for specific weights
def fc1_hook(grad):
    grad_clone = grad.clone()
    grad_clone[fc1_map == 0] = 0
    return grad_clone

def conv1_hook(grad):
    grad_clone = grad.clone()
    grad_clone[conv1_map == 0] = 0
    return grad_clone

def conv2_hook(grad):
    grad_clone = grad.clone()
    grad_clone[conv2_map == 0] = 0
    return grad_clone


# Register the hook for the specific parameter
hook1 = model.fc1.weight.register_hook(fc1_hook)
hook2 = model.conv1.weight.register_hook(conv1_hook)
hook3 = model.conv2.weight.register_hook(conv2_hook)

#train
def train(dataloader, model, loss_fn, optimizer,scheduler):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        myloss= CustomCrossEntropyLoss(pred,y)
        #print("pytorch Loss:",loss)
        #print("my loss:",myloss)
        optimizer.zero_grad()
        #loss
        myloss.backward()
        
        optimizer.step()
        if batch % 400 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    scheduler.step()    


#test
def test(dataloader, model,print_results=True):
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
    if(print_results):
        print(f"Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")
    return 100*correct, test_loss

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

#save the starting errors
starting_accuracy_forgotten,_=test(test_only_forgotten_dataloader, model,False)
starting_accuracy_new,_=test(test_only_to_learn_dataloader, model,False)





#train and test
for t in range(epochs_forget):
    print(f"Epoch {t+1}\n-------------------------------")
    print("Forgotting...")
    train(train_only_forgotten_dataloader, model, loss_fn, optimizer,scheduler)
    print("Accuracy on dataset:")
    test(test_to_learn_dataloader, model)
    print("Accuracy on forgotten data:")
    test(test_only_forgotten_dataloader, model)
    print("Accuracy on the new data:")
    test(test_only_to_learn_dataloader, model)

#train and test
for t in range(epochs_relearn):
    print(f"Epoch {t+1}\n-------------------------------")
    print("ReLearning...")
    train(training_to_learn_dataloader, model, loss_fn, optimizer,scheduler)
    print("Accuracy on dataset:")
    test(test_to_learn_dataloader, model)
    print("Accuracy on forgotten data:")
    test(test_only_forgotten_dataloader, model)
    print("Accuracy on the new data:")
    test(test_only_to_learn_dataloader, model)


hook1.remove()
hook2.remove()
hook3.remove()
print("\n\n")
print("Final error:")
test(test_to_learn_dataloader, model)
print("Final error on forgotten data:")
test(test_only_forgotten_dataloader, model)
print("Final error on the new data:")
test(test_only_to_learn_dataloader, model)
print("Starting accuracy on forgotten data:\n",starting_accuracy_forgotten)
print("Starting accuracy on the new data:\n",starting_accuracy_new)


#save the model
torch.save(model.state_dict(), "modelRetr.pth")



#print(test_data[0].shape)