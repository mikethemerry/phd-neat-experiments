import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pmlb import fetch_data

class DenseNet(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, dropout_p=0.5):
        super(DenseNet, self).__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(hidden_layers[-1], output_size)
        self.dropout_p = dropout_p
        
    def forward(self, x):
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.output(x)
        return x
    
    def train_model(self, x, y, epochs, validation_data, patience=3):
        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.parameters(), lr=0.01)
        validation_loss_min = float('inf')
        stop_epoch = 0
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                validation_loss = criterion(self(validation_data[0]), validation_data[1])
                if validation_loss <= validation_loss_min:
                    stop_epoch = 0
                    validation_loss_min = validation_loss
                    best_weights = self.state_dict()
                else:
                    stop_epoch += 1
                if stop_epoch == patience:
                    self.load_state_dict(best_weights)
                    break
        return self


# Fetch a binary classification dataset from PMLB
X, y = fetch_data('diabetes', return_X_y=True)

# Convert to PyTorch tensors and split into training and validation sets
inputs = torch.tensor(X, dtype=torch.float32)
targets = torch.tensor(y, dtype=torch.float32)

val_inputs = inputs[-100:]
val_targets = targets[-100:]
inputs = inputs[:-100]
targets = targets[:-100]

# Create an instance of the dense neural network class
model = DenseNet(input_size=inputs.shape[1],
                #   hidden_sizes=[32, 16],
                hidden_layers=[32, 16],
                  output_size=1)

# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the model
# model.train(inputs, targets, val_inputs, val_targets, criterion, optimizer, epochs=100)


# Make predictions on the validation set
with torch.no_grad():
    outputs = model(val_inputs)
    predictions = (outputs > 0.5).float()


# Calculate the accuracy of the predictions
accuracy = (predictions == val_targets).float().mean()
print(f"Validation accuracy: {accuracy:.4f}")


model.train_model(inputs, targets, 5000, (val_inputs,val_targets))

# Make predictions on the validation set
with torch.no_grad():
    outputs = model(val_inputs)
    predictions = (outputs > 0.5).float()

# Calculate the accuracy of the predictions
accuracy = (predictions == val_targets).float().mean()
print(f"Validation accuracy: {accuracy:.4f}")
