import argparse
import os
import json

from explaneat.experimenter.experiment import GenericExperiment
from explaneat.data.wranglers import GENERIC_WRANGLER
from explaneat.experimenter.results import Result

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

parser = argparse.ArgumentParser(description="Provide the experiment config")
parser.add_argument('conf_file',
                    metavar='experiment_config_file',
                    type=str,
                    help="Path to experiment config")
parser.add_argument("ref_file",
                    metavar='experiment_reference_file',
                    type=str,
                    help="Path to experiment ref file")

args = parser.parse_args()

experiment = GenericExperiment(
    args.conf_file,
    confirm_path_creation=False,
    ref_file=args.ref_file)
logger = experiment.logger


experiment.create_logging_header("Starting {}".format(__file__), 50)
model_config = experiment.config['model']['neural_network']

# ------------------- Set up environment ------------------------------


USE_CUDA = torch.cuda.is_available()
USE_CUDA = False
device = torch.device("cuda:1" if USE_CUDA else "cpu")

# ---------------- Load data ------------------------------

processed_data_location = experiment.data_folder

generic_wrangler = GENERIC_WRANGLER(processed_data_location)

X_train, y_train = generic_wrangler.train_sets
X_test, y_test = generic_wrangler.test_sets


train_data = generic_wrangler.Train_Dataset
train_loader = DataLoader(train_data,
                          batch_size=model_config["batch_size"],
                          shuffle=True)

# validate_data = generic_wrangler.Test_Dataset
# validate_loader = DataLoader(dataset=validate_data,
#                              batch_size=model_config["batch_size"],
#                              shuffle=True)

total_step = len(train_loader)

def train_val_split(xs, ys, val_frac=0.1, shuffle=True, seed=0, to_float=True):
    """
    Splits a PyTorch tensor into training and validation sets.
    """
    # Set the seed for reproducibility
    torch.manual_seed(seed)
    
    # Get the number of samples
    num_samples = xs.size()[0]
    
    # Shuffle the data if desired
    if shuffle:
        perm = torch.randperm(num_samples)
        xs = xs[perm]
        ys = ys[perm]
    
    # Calculate the number of samples in the validation set
    num_val = int(num_samples * val_frac)
    
    # Split the data into training and validation sets
    xs_train = xs[:-num_val]
    ys_train = ys[:-num_val]
    xs_val = xs[-num_val:]
    ys_val = ys[-num_val:]
    
    if to_float:
        xs_train = xs_train.float()
        ys_train = ys_train.float()
        xs_val = xs_val.float()
        ys_val = ys_val.float()

    return xs_train, ys_train, xs_val, ys_val


xsnn = torch.tensor(X_train.values).to(device)
ysnn = torch.tensor(y_train.values).to(device)

X_train, y_train, X_val, y_val = train_val_split(xsnn, ysnn, seed=experiment.config["random_seed"])


# ------------------- Define model ------------------------------


class DenseNet(nn.Module):
    ## From ChatGPT
    def __init__(self, input_size, hidden_layers, output_size, dropout_p=0.25):
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
        x = torch.sigmoid(x)
        return x    
    
    def train(self, x, y, epochs, validation_data, criterion=nn.BCELoss(), patience=50):
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

class NeuralNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_width=64):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_width)
        self.fc2 = nn.Linear(hidden_width, hidden_width)
        self.fc3 = nn.Linear(hidden_width, hidden_width)
        self.fc4 = nn.Linear(hidden_width, hidden_width)
        self.fc5 = nn.Linear(hidden_width, output_size)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.fc5(out)
        return out


# ------------------- prepare early stopping  ------------------------

class EarlyStopping():
    def __init__(self,
                 tolerance=5,
                 reset_tolerance=10,
                 min_delta=0):
        """Early stopping method that stops after tolerance, but if training
        reengages for reset_tolerance it continues

        Args:
            tolerance (int, optional): number of epochs without progress before 
            stopping. Defaults to 5.
            reset_tolerance (int, optional): number of epochs of improvement after
            to reset tolerance to 0. Defaults to 10.
            min_delta (float, optional): minimum amount of improvement to be called
            improvement. Defaults to 0.
        """
        self.train_losses = []
        self.validation_losses = []

        self.tolerance = tolerance
        self.reset_tolerance = reset_tolerance
        self.min_delta = min_delta

        self.counter = 0
        self.reset_counter = 0
        self.early_stop = False

    def __call__(self,
                 train_loss,
                 validation_loss):
        self.train_losses.append(train_loss)
        self.validation_losses.append(validation_loss)

        if (validation_loss - train_loss) > self.min_delta:
            self.counter += 1
            self.reset_counter = 0
            if self.counter >= self.tolerance:
                self.early_stop = True
        else:
            self.reset_counter += 1
            if self.reset_counter >= self.reset_tolerance:
                self.counter = 0
                self.reset_counter = 0


# ------------------- instantiate model ------------------------------

# nn_model = NeuralNet(generic_wrangler.input_size,
                    #  generic_wrangler.output_size).to(device)

# criterion = nn.BCEWithLogitsLoss().to(device)
# optimizer = torch.optim.Adam(
    # nn_model.parameters(), lr=model_config['learning_rate'])

model = DenseNet(generic_wrangler.input_size,
                [32, 64, 64, 32],
                generic_wrangler.output_size).to(device)

# ------------------- train model ------------------------------

model.train(X_train, y_train, model_config['num_epochs'], (X_val, y_val))

# for epoch in range(model_config['num_epochs']):

#     # for i, (xsnn, ysnn) in enumerate(train_loader):
#     #     # Move tensors to the configured device
#     #     xsnn = xsnn.float().to(device)
#     #     ysnn = ysnn.view(-1, 1).float().to(device)
#     xsnn = torch.tensor(X_train).to(device)
#     ysnn = torch.tensor(y_train).to(device)

#     # Forward pass
#     outputs = nn_model(xsnn)
#     train_loss = criterion(outputs, ysnn)

#     # Backward and optimize
#     optimizer.zero_grad()
#     train_loss.backward()
#     optimizer.step()
#     if (epoch+1) % 50 == 0:
#         print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
#               .format(epoch+1, model_config['num_epochs'], i+1, total_step, train_loss.item()))

# ------------------- get predictions ------------------------------

nn_preds = torch.sigmoid(model.forward(torch.from_numpy(
    X_test.to_numpy()).float().to(device)).to(device)).detach().numpy()

nn_preds = [float(pred[0]) for pred in nn_preds]

preds_results = Result(
    json.dumps(list(nn_preds)),
    "nn_prediction",
    experiment.config['experiment']['name'],
    experiment.config['data']['raw_location'],
    experiment.experiment_sha,
    0,
    {
        "iteration": 0
    }
)
experiment.results_database.add_result(preds_results)

experiment.create_logging_header("Ending {}".format(__file__), 50)
