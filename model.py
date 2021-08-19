import torch
import torch.nn as nn

# Using PyTorch for deep learning to implement a feed forward neural net, with 2 hidden layers
# Bag of words is the input to the neural net, one layer is connected using the number of patterns as the input size
# There are then two hidden layers used in the neural net
# The output is the number of classes
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        # Creating 3 linear layers structure is Linear(input, output)
        self.l1 = nn.Linear(input_size, hidden_size) # Input size is the input
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes) # Number of classes is the output
        # Input size and number of clases are fixed, but can change hidden size
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x) # Apply first linear layer
        out = self.relu(out) # Apply activation function
        # Repeat adding linear layers and activation functions
        out = self.l2(out) 
        out = self.relu(out)
        out = self.l3(out)
        # No activation and no softmax

        return out

