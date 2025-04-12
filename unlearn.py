import torch
import torchvision.datasets as datasets
import torch.nn as nn
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

def unlearn(forget_target=4,sub_target=9,learning_rate=4e-3,batch_size=16,epochs_forget=2,epochs_relearn=10,grad_map2=[8,32,2000]):

    '''
    Compute and return the gradients of the weights of a specific layer for a specific class and input image
    '''
    def compute_gradients(model,layer,inputs, target_class):
        model.eval()
        inputs.requires_grad = True
        # Zero the gradients of the weights
        model.zero_grad()
        inputs = inputs.to(device)
        outputs = model(inputs)
        target_class_tensor = torch.tensor([target_class]).to(device)  # Convert the target class to a tensor
        outputs = outputs.gather(1, target_class_tensor.view(-1, 1)).squeeze()  # Get the output of the target class
        outputs.backward()

        weight_gradients = layer.weight.grad.squeeze()

        return weight_gradients
    
    '''
    Visualize the gradients of the weights
    '''
    def show_gradient_map(weight_gradients):
        plt.imshow(weight_gradients.detach().numpy(), cmap=plt.cm.gray)
        plt.axis('off')
        plt.show()

    '''
    Calculate the gradient map and treshold based on the top k gradients. Than zero out the rest.
    '''
    def calculate_map_and_treshold(weight_gradients,k=2000):
        grad_map= weight_gradients.clone().detach()

        #take the value of the top k highest gradients
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


    # Load the model
    model = CNN()

    #WARNING: IF YOU WANT TO TEST THE CODE, PLEASE LOAD THE MODEL WITH THE CORRECT NAME
    model.load_state_dict(torch.load("models/modelNo0.pth",map_location=torch.device(device)))

    model=model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    ################################# Utility functions part #################################

    def log_softmax(x):
        return x - torch.logsumexp(x,dim=1, keepdim=True)

    def CustomCrossEntropyLoss(outputs, targets):
        epsilon=1e-6
        num_examples = targets.shape[0]
        batch_size = outputs.shape[0]
        outputs[targets==forget_target]=-outputs[targets==forget_target]
        outputs = log_softmax(outputs+epsilon)
        
        #take only the target loss
        outputs = outputs[range(batch_size), targets]

        return - torch.sum(outputs)/num_examples

    def train(dataloader, model, loss_fn, optimizer,scheduler):
        size = len(dataloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            #loss = loss_fn(pred, y)
            myloss= CustomCrossEntropyLoss(pred,y)
            optimizer.zero_grad()
            #loss
            myloss.backward()
            
            optimizer.step()
            if batch % 400 == 0:
                myloss, current = myloss.item(), batch * len(X)
                #print(f"loss: {myloss:>7f}  [{current:>5d}/{size:>5d}]")
        scheduler.step()    



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


    ################################# Dataset preprocessing part #################################

    # Load and preprocess the dataset.
    print("Processing dataset...")
    #this will contain only the train data about the forgotten class
    train_only_forgotten_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )
    train_mask = train_only_forgotten_data.targets == forget_target
    train_only_forgotten_data.data = train_only_forgotten_data.data[train_mask]
    train_only_forgotten_data.targets = train_only_forgotten_data.targets[train_mask]
    train_only_forgotten_dataloader = DataLoader(train_only_forgotten_data, batch_size=batch_size)



    #this will contain only the train data about the new class
    train_only_to_learn = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )
    train_mask = train_only_to_learn.targets == sub_target
    train_only_to_learn.data = train_only_to_learn.data[train_mask]
    train_only_to_learn.targets = train_only_to_learn.targets[train_mask]
    train_only_to_learn_dataloader = DataLoader(train_only_to_learn, batch_size=batch_size)

    #this will contain only the test data about the new class
    test_only_to_learn = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    test_mask = test_only_to_learn.targets == sub_target
    test_only_to_learn.data = test_only_to_learn.data[test_mask]
    test_only_to_learn.targets = test_only_to_learn.targets[test_mask]
    test_only_to_learn_dataloader = DataLoader(test_only_to_learn, batch_size=batch_size)


    #this will contain only the test data about the forgotten class
    test_only_forgotten_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )
    test_mask = test_only_forgotten_data.targets == forget_target
    test_only_forgotten_data.data = test_only_forgotten_data.data[test_mask]
    test_only_forgotten_data.targets = test_only_forgotten_data.targets[test_mask]
    test_only_forgotten_dataloader = DataLoader(test_only_forgotten_data, batch_size=batch_size)




    #this will contain the test data where the forget class is removed
    test_to_learn = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )
    test_mask = test_to_learn.targets != forget_target
    test_to_learn.data = test_to_learn.data[test_mask]
    test_to_learn.targets = test_to_learn.targets[test_mask]
    test_to_learn_dataloader = DataLoader(test_to_learn, batch_size=batch_size)

    #this will contain the test data where the new class and the forgotten class are removed.
    test_static = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )
    test_mask = (test_static.targets != sub_target) & (test_static.targets != forget_target)
    test_static.data = test_static.data[test_mask]
    test_static.targets = test_static.targets[test_mask]
    test_static_dataloader = DataLoader(test_static, batch_size=batch_size)



    ################################# Forget Gradient computation part #################################

    #create the gradient holders
    grads_conv1 = torch.zeros(model.conv1.weight.shape).squeeze().to(device)
    grads_conv2 = torch.zeros(model.conv2.weight.shape).to(device)
    grads_fc1 = torch.zeros(model.fc1.weight.shape).squeeze().to(device)

    #grads_fc2 = torch.zeros(model.fc2.weight.shape).squeeze().to( device)


    # Compute the gradients of the weights of all layers for the target class
    for img,_ in train_only_forgotten_dataloader:
        #img = img.unsqueeze(0)
        grads_conv1 += compute_gradients(model, model.conv1, img, forget_target).abs()
        grads_conv2 += compute_gradients(model, model.conv2, img, forget_target).abs()
        grads_fc1 += compute_gradients(model, model.fc1, img, forget_target).abs()
        #grads_fc2 += compute_gradients(model, model.fc2, img, forget_target).abs()


    #takes about 10% of the highest gradients
    conv1_map,_=calculate_map_and_treshold(grads_conv1,4)
    conv1_map=conv1_map.unsqueeze(1)
    conv2_map,_=calculate_map_and_treshold(grads_conv2,16)
    fc1_map,_=calculate_map_and_treshold(grads_fc1,1000)
    #fc2_map,_=calculate_map_and_treshold(grads_fc2,10)




    ################################# Forgetting part #################################


    # Define a custom backward hook to zero out gradients for specific weights
    def fc1_hook(grad):
        grad_clone = grad.clone()
        grad_clone[fc1_map == 0] = 0
        return grad_clone

    #def fc2_hook(grad):
    #    grad_clone = grad.clone()
    #    grad_clone[fc2_map == 0] = 0
    #    return grad_clone

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
    #hook2 = model.fc2.weight.register_hook(fc2_hook)
    hook3 = model.conv1.weight.register_hook(conv1_hook)
    hook4 = model.conv2.weight.register_hook(conv2_hook)


    #save the starting errors
    starting_accuracy_forgotten,_=test(test_only_forgotten_dataloader, model,False)
    starting_accuracy_new,_=test(test_only_to_learn_dataloader, model,False)





    #train and test
    for t in range(epochs_forget):
        #print(f"Epoch {t+1}\n-------------------------------")
        print("Forgetting...")
        train(train_only_forgotten_dataloader, model, loss_fn, optimizer,scheduler)
        print("Accuracy on dataset:")
        test(test_to_learn_dataloader, model)
        print("Accuracy on forgotten data:")
        test(test_only_forgotten_dataloader, model)
        print("Accuracy on the new data:")
        test(test_only_to_learn_dataloader, model)

    hook1.remove()
    #hook2.remove()
    hook3.remove()
    hook4.remove()

    ################################# Relearn Gradient computation part #################################

    #create the gradient holders
    grads_conv1 = torch.zeros(model.conv1.weight.shape).squeeze().to(device)
    grads_conv2 = torch.zeros(model.conv2.weight.shape).to(device)
    grads_fc1 = torch.zeros(model.fc1.weight.shape).squeeze().to(device)
    #grads_fc2 = torch.zeros(model.fc2.weight.shape).squeeze().to(device)

    # Compute the gradients of the weights of all layers for the target class
    for img,_ in train_only_to_learn_dataloader:
        #img = img.unsqueeze(0)
        grads_conv1 +=compute_gradients(model, model.conv1, img, sub_target).abs()
        grads_conv2 += compute_gradients(model, model.conv2, img, sub_target).abs()
        grads_fc1 += compute_gradients(model, model.fc1, img, sub_target).abs()
        #grads_fc2 += compute_gradients(model, model.fc2, img, sub_target).abs()

    #takes about 10% of the highest gradients
    conv1_map,_=calculate_map_and_treshold(grads_conv1,grad_map2[0]) #4
    conv1_map=conv1_map.unsqueeze(1)
    conv2_map,_=calculate_map_and_treshold(grads_conv2,grad_map2[1]) #16
    fc1_map,_=calculate_map_and_treshold(grads_fc1,grad_map2[2 ]) #1000
    #fc2_map,_=calculate_map_and_treshold(grads_fc2,80)

    ################################# Relearning part #################################

    # Define a custom backward hook to zero out gradients for specific weights
    def fc1_hook(grad):
        grad_clone = grad.clone()
        grad_clone[fc1_map == 0] = 0
        return grad_clone

    #def fc2_hook(grad):
    #    grad_clone = grad.clone()  
    #    grad_clone[fc2_map == 0] = 0
    #    return grad_clone

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
    #hook2 = model.fc2.weight.register_hook(fc2_hook)
    hook3 = model.conv1.weight.register_hook(conv1_hook)
    hook4 = model.conv2.weight.register_hook(conv2_hook)

    print("\n\n")
    for t in range(5):
        print(f"Epoch {t+1}\n-------------------------------")
        print("Learning the new class....  epoch",t+1)
        train(train_only_to_learn_dataloader, model, loss_fn, optimizer,scheduler)
        print("Accuracy on dataset:")
        test(test_to_learn_dataloader, model)
        print("Accuracy on forgotten data:")
        test(test_only_forgotten_dataloader, model)
        print("Accuracy on the new data:")
        test(test_only_to_learn_dataloader, model)
        print("Accuracy on static data:")
        test(test_static_dataloader,model)
        print("\n")

    hook1.remove()
    #hook2.remove()
    hook3.remove()
    hook4.remove()


    print("\n\n Results with paramers",sub_target,",",forget_target,",",learning_rate,",",grad_map2,":")
    print("On dataset:")
    dataset_score=test(test_to_learn_dataloader, model)
    print("On forgotten data:")
    forgotten_score=test(test_only_forgotten_dataloader, model)
    print("On the new data:")
    new_data_score=test(test_only_to_learn_dataloader, model)
    print("On static data:")
    static_score=test(test_static_dataloader,model)
    print("Starting accuracy on forgotten data:\n",starting_accuracy_forgotten)
    print("Starting accuracy on the new data:\n",starting_accuracy_new)

    return model,dataset_score,forgotten_score,new_data_score,static_score,starting_accuracy_forgotten

    #save the model
    #torch.save(model.state_dict(), "models/modelRetr.pth")
