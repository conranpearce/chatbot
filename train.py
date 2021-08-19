import numpy as np
import json
from nltk_utils import tokenize, stem, bag_of_words
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet

# Execute train.py before running app.py, to train the training data

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = [] 
FILE = "data.pth"

# Putting the NLP together and training the model

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)

    for pattern in intent['patterns']:
        w = tokenize(pattern) # Tokenize pattern
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '!', '.', ','] # Ignore punctuation
all_words = [stem(w) for w in all_words if w not in ignore_words] # Carry out stemming, if word is not included in the ignore_words array
# Sort words and convert into a set, so that only unique words are included
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# Create training data, x_train and y_train correspond to the data set and label of data set
x_train = []
y_train = [] 

# Use the result of the bag of words for the x training set (which has also undergone tokenisation and stemming)
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words) # Create a bag of words
    x_train.append(bag) 

    # Append labels for training data
    label = tags.index(tag)
    y_train.append(label) # Using CrossEntropyLoss, so only needing the label of the pattern

# Using numpy for training data
x_train = np.array(x_train)
y_train = np.array(y_train)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# Setting parameters for the training size and learning rate for training the data set
batch_size = 8
hidden_size = 8
output_size = len(tags) # Number of classes, as the output
input_size = len(x_train[0])
learning_rate = 0.001
num_epochs = 1000

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2) # Using batch training

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)

        # Backward pass and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Every 100 step print the epoch
    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss={loss.item():.4f}')

# Express to the user the final loss
print(f'Final Loss, Loss={loss.item():.4f}')
  
# Saving the data after training 
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}

# Saving the trained file
torch.save(data, FILE)
print(f'Training is completed. File has been saved to {FILE}')
