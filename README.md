# Machine unlearning
## About the project
This is a project for the Deep Learning and Artificial Intelligence course at Sapienza University based on the topic 1 of the [project list](https://raw.githubusercontent.com/erodola/DLAI-s2-2024/main/project_list.pdf).

This project explores a simple approach to machine unlearning through a two-phase process. 
In the former, the model selectively forgets a designated class, and in the second phase, it learns a new class. In both phases it uses only data relevant to these two classes. 
The goal is to achieve this using minimal data and affecting only a small subset of the model’s weights.
In this case a simple CNN has been used.

## The content

- main.py: Example usage of the unlearn function
- unlearn.py: Contains the unlearn function. This is where the unlearn and relearn happen.
- trainer.py: All the model used in this project are been trained with this python script.
- modelNoX.pth: pretrained CNN model that has not been trained on the class 'X'.

## How to use

If you want to train a new model, change the 'target' variable in trainer.py to the desired class that you want to exclude from the training.

If you want to try to replicate the results and use the unlearn method, it's essential that you change the loaded pretrained model to the right one. 
To do this change, at line 80, "modelNo9.pth" with the right one.
```
#WARNING: IF YOU WANT TO TEST THE CODE, PLEASE LOAD THE MODEL WITH THE CORRECT NAME
model.load_state_dict(torch.load("models/modelNo0.pth",map_location=torch.device(device)))
```

Then, call 
```
unlearn(class_to_forget,new_class_to_learn);
```

