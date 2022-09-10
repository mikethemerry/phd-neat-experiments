import argparse
import os

from explaneat.experimenter.experiment import GenericExperiment
from explaneat.data.wranglers import GENERIC_WRANGLER
from explaneat.experimenter.results import Result

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


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

# ---------------- Load data ------------------------------

processed_data_location = experiment.data_folder

generic_wrangler = GENERIC_WRANGLER(processed_data_location)

X_train, y_train = generic_wrangler.train_sets
X_test, y_test = generic_wrangler.test_sets


train_data = generic_wrangler.Train_Dataset
train_loader = DataLoader(train_data,
                          batch_size=model_config["batch_size"],
                          shuffle=True)

validate_data = generic_wrangler.Test_Dataset
validate_loader = DataLoader(dataset=validate_data,
                             batch_size=model_config["batch_size"],
                             shuffle=True)

total_step = len(train_loader)

# ------------------- Set up environment ------------------------------


USE_CUDA = torch.cuda.is_available()
USE_CUDA = False
device = torch.device("cuda:1" if USE_CUDA else "cpu")


# ------------------- Define model ------------------------------


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


# ------------------- instantiate model ------------------------------


nn_model = NeuralNet(generic_wrangler.input_size,
                     generic_wrangler.output_size).to(device)

criterion = nn.BCEWithLogitsLoss().to(device)
optimizer = torch.optim.Adam(
    nn_model.parameters(), lr=model_config['learning_rate'])

# ------------------- train model ------------------------------

for epoch in range(model_config['num_epochs']):
    for i, (xsnn, ysnn) in enumerate(train_loader):
        # Move tensors to the configured device
        xsnn = xsnn.float().to(device)
        ysnn = ysnn.view(-1, 1).float().to(device)

        # Forward pass
        outputs = nn_model(xsnn)
        train_loss = criterion(outputs, ysnn)

        # Backward and optimize
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        if (epoch+1) % 50 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, model_config['num_epochs'], i+1, total_step, train_loss.item()))

# ------------------- get predictions ------------------------------

nn_preds = torch.sigmoid(nn_model.forward(torch.from_numpy(
    X_test.to_numpy()).float().to(device)).to(device)).detach().numpy()

nn_preds = [pred[0] for pred in nn_preds]

preds_results = Result(
    nn_preds,
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
