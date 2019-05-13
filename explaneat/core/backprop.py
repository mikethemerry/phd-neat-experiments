import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random
import pandas as pd
import numpy as np

import copy



def tt(num):
    return nn.Parameter(torch.tensor([float(num)], requires_grad=True))


def neatSigmoid(num):
    return torch.sigmoid(4.9*num)

class NeatNet():
    def __init__(self, genome, config):
        # super(NeatNet, self).__init__()
        self._modules = []
        self.config = config
        self.genome = genome
        self.connections = {
            k: tt(connection.weight) for k, connection in genome.connections.items()
        }

        self.biases = {
            k: tt(node.bias) for k, node in genome.nodes.items()
        }

        self.params = []
        for k, v in self.connections.items():
            self.params.append(v)
        for k, v in self.biases.items():
            self.params.append(v)

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

        self.optimizer = optim.Adadelta(self.params, lr=1.5)
        self.criterion = nn.BCELoss()




    def get_order_of_nodes(self):
        order_of_nodes = []
        nodes_to_propagate = [
            k for k in self.input_keys
        ]
        connOuts = copy.deepcopy(self.connections_by_output)

        while len(nodes_to_propagate) > 0:
            
            node = nodes_to_propagate.pop(0)
            order_of_nodes.append(node)
            # handle nodes with no output
            if node not in self.connections_by_input:
                order_of_nodes.append(node)
                continue
            for connection in self.connections_by_input[node]:
                

                del connOuts[connection[1]][connection]

                if len(connOuts[ connection[1] ] ) == 0:
                    if connection[1] not in self.config.genome_config.output_keys:
                        nodes_to_propagate.append(connection[1])
                    else:
                        order_of_nodes.append(connection[1])
        for outputNode in self.config.genome_config.output_keys:
            if outputNode not in order_of_nodes:
                order_of_nodes.append(outputNode)
        return order_of_nodes

    def create_parameters(self):
        assert self.order_of_nodes is not None

    def activateNode(self, node):
        vals = [
        ]
        # handler for if the output is not connected
        if node in self.connections_by_output:
            for connection in self.connections_by_output[node]:
                # Handler for if there is a node not prior connected to the
                # inputs
                if connection[0] in self.order_of_nodes:
                    vals.append(self.nodeVals[connection[0]] * self.connections[connection])

        vals.append(torch.tensor(self.genome.nodes[node].bias, requires_grad=True))
        mySum = sum(vals)
        return torch.sigmoid(5.0*mySum)

    def forward(self, inputs):
        next_steps = []

                # h1 = torch.sigmoid(4.9*(x1*self.g1 + x2*self.g2 + self.b1))
        
        self.nodeVals = {}
        for k, inputVal in enumerate(inputs):
            self.nodeVals[self.input_keys[k]] = inputVal
        
        for node in self.order_of_nodes:
            # pass over input nodes
            if node in self.nodeVals:
                continue
            self.nodeVals[node] = self.activateNode(node)


        return self.nodeVals[self.output_keys[0]]


    def optimise(self, xs, ys, nEpochs = 100):
        if not type(xs) is torch.Tensor:
            xs = torch.tensor(xs)
        if not type(ys) is torch.Tensor:
            ys = torch.tensor(ys)

        # print('going to train for %s epochs' % nEpochs)
        for epoch in range(nEpochs):
            for inX in range(len(xs)):
                self.optimizer.zero_grad()   # zero the gradient buffers

                output = self.forward(xs[inX])
                target = ys[inX]
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
    optimize = optimise

    def meanLoss(self, xs, ys):
        if not type(xs) is torch.Tensor:
            xs = torch.tensor(xs)
        if not type(ys) is torch.Tensor:
            ys = torch.tensor(ys)
        losses = []
        for inX in range(len(xs)):

            output = self.forward(xs[inX])
            target = ys[inX]
            losses.append(self.criterion(output, target))
        return sum(losses)/len(losses)

    def updateGenomeWeights(self, genome):
        """takes in GenomeClass object and replaces the genome weights in place
        """
        for k in genome.connections:
            genome.connections[k].weight = float(self.connections[k][0])
        for k in genome.nodes:
            genome.nodes[k].bias = float(self.biases[k][0])
        




class Net(torch.nn.Module):

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



# for gen in range(100):

#     for inX in range(4):
#         optimizer.zero_grad()   # zero the gradient buffers

#         output = net.forward(inputs[inX])
#         target = outputs[inX]
#         loss = criterion(output, target)
#         loss.backward()
#         optimizer.step()
#     if gen%2000 == 1:
#         print(gen)
#         for inX in range(30):

#             output = net.forward(inputs[inX])
#             target = outputs[inX]

#             print(output, target)
# for inX in range(len(inputs)):
#     input = inputs[inX]
#     output = net.forward(input)
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