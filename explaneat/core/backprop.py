import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random
import pandas as pd
import numpy as np

import copy



def tt(num):
    return nn.Parameter(torch.randn(1, requires_grad=True))
    # return nn.Parameter(torch.tensor([float(num)], requires_grad=True))

def neatSigmoid(num):
    return torch.sigmoid(4.9*num)

class NeatNet():
    def __init__(self, config, genome):
        # super(NeatNet, self).__init__()
        self._modules = []
        self.config = config
        self.genome = genome
        self.connections = {
            k: tt(connection.weight) for k, connection in genome.connections.items()
        }

        self.input_keys = self.config.genome_config.input_keys
        self.output_keys = self.config.genome_config.output_keys

        # self.connections_by_input = {
        #     ks[0] : {
        #         k: c for k, c in genome.connections.items() if k[0] == ks[0]
        #     } for ks in genome.connections
        # }

        # self.connections_by_output = {
        #     ks[1] : {
        #         k: c for k, c in genome.connections.items() if k[1] == ks[1]
        #     } for ks in genome.connections
        # }

        self.connections_by_input = {}
        self.connections_by_output = {}
        for k, c in genome.connections.items():
            if not k[0] in self.connections_by_input.keys():
                self.connections_by_input[k[0]] = {}
            self.connections_by_input[k[0]][k] = c
            if not k[1] in self.connections_by_output.keys():
                self.connections_by_output[k[1]] = {}
            self.connections_by_output[k[1]][k] = c
        self.order_of_nodes = self.get_order_of_nodes()


    def get_order_of_nodes(self):
        order_of_nodes = []
        nodes_to_propagate = [
            k for k in self.input_keys
        ]
        connOuts = copy.deepcopy(self.connections_by_output)

        while len(nodes_to_propagate) > 0:
            
            node = nodes_to_propagate.pop(0)
            order_of_nodes.append(node)
            
            for connection in self.connections_by_input[node]:
                

                del connOuts[connection[1]][connection]

                if len(connOuts[ connection[1] ] ) == 0:
                    if connection[1] not in self.config.genome_config.output_keys:
                        nodes_to_propagate.append(connection[1])
                    else:
                        order_of_nodes.append(connection[1])
        return order_of_nodes

        

    def forward(self, inputs):
        next_steps = []

        return output


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.g1 = tt(-3.79)
        self.g2 = tt(-3.32)
        self.g3 = tt(-2.26)
        self.g4 = tt(4.72)
        self.g5 = tt(1.57)
        self.b1 = tt(-5.24)
        self.b2 = tt(-0.07)


    def forward(self, x):
        x1= x[0]
        x2 = x[1]
        
        h1 = torch.sigmoid(4.9*(x1*self.g1 + x2*self.g2 + self.b1))

        o1 = torch.sigmoid(4.9*(x1*self.g3 + h1*self.g4 + x2*self.g5 + self.b2))

        return o1



# net = Net()
# print(net)

# # optimizer = optim.SGD(net.parameters(), lr=0.01)
# optimizer = optim.Adadelta(net.parameters())
# # criterion = nn.MSELoss()
# criterion = nn.BCELoss()


def xor(a, b):
    response = False
    if a > 0.5 and b < 0.5:
        response = True
    if a < 0.5 and b > 0.5:
        response = True
    # return (1.0, 0.0) if response else (0.0, 1.0)
    return 1.0 if response else 0.0

def create_n_points(n, size, min=0.0, max=1.0):
    data = []
    for _ in range(n):
        data.append(tuple([
            random.uniform(min, max) for ii in range(size)
        ]))

    return data

# correct solution:
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

def overUnder(val, threshold):
    return 1. if val > threshold else 0


# xor_inputs_2 = create_n_points(400, 2)

# xor_outputs_2 = [
#     tuple( [xor(tup[0], tup[1])] ) for tup in xor_inputs_2
# ]



# # 2-input XOR inputs and expected outputs.
# # xor_inputs = torch.tensor([(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)])
# # xor_outputs = torch.tensor([   (0.0,),     (1.0,),     (1.0,),     (0.0,)])

# # output = net(xor_inputs_2[0])
# # target = xor_outputs[0]
# # loss = criterion(output, target)

# inputs = torch.tensor(xor_inputs_2)
# outputs = torch.tensor(xor_outputs_2)



# for gen in range(5000):
    
#     for inX in range(4):
#         optimizer.zero_grad()   # zero the gradient buffers

#         output = net(inputs[inX])
#         target = outputs[inX]
#         loss = criterion(output, target)
#         loss.backward()
#         optimizer.step()
#     if gen%2000 == 1:
#         print(gen)
#         for inX in range(30):

#             output = net(inputs[inX])
#             target = outputs[inX]

#             print(output, target)
# for inX in range(len(inputs)):
#     input = inputs[inX]
#     output = net(input)
#     target = outputs[inX]

#     print(input, output, target)

# for p in net.parameters():
#     print(p)

# results = []
# for xi, xo in zip(inputs, outputs):
#     output = net(xi)
#     # print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))
#     results.append([xi[0].numpy(), xi[1].numpy(), output[0].detach().numpy()])

# df = pd.DataFrame(results)
# df.to_csv('./results_graddesc_2.csv')