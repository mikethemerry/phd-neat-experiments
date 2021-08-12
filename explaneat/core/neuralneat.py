
import pandas as pd
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim


LAYER_TYPE_CONNECTED = "CONNECTED"
LAYER_TYPE_INPUT = "INPUT"
LAYER_TYPE_OUTPUT = "OUTPUT"

class NeuralNeat(nn.Module):
    ## Creates a PyTorch Neural Network from an ExplaNEAT genome
    def __init__(self, genome, config, criterion=nn.BCELoss(), optimiser = optim.Adadelta):
        super(NeuralNeat, self).__init__()  # just run the init of parent class (nn.Module)
        self.genome = genome
        layers = self.parse_genome_to_layers(genome, config)
        self.layers = layers
        self.weights = {layer_id: self._tt(layer['input_weights'].copy()) for layer_id, layer in layers.items()}
        self.biases = {layer_id: self._tt(layer['bias'].copy()) for layer_id, layer in layers.items()}
        self.layer_types = {layer_id: layer['layer_type'] for layer_id, layer in layers.items()}
        self.layer_inputs = {layer_id: layer['input_layers'] for layer_id, layer in layers.items()}
        self.n_layers = len(layers)

        self._outputs = None

        for w_id, w in self.weights.items():
            self.register_parameter(name ="weight_%s"%w_id, param=w)

        for b_id, b in self.biases.items():
            self.register_parameter(name = "bias_%s"%b_id, param=b)

        self.criterion = criterion
        self.optimiser = optimiser
        self.optimizer = self.optimiser

        # x = F.relu(self.fc1(x))

        # self.fc2 = nn.Linear(512, 10)

    def parse_genome_to_layers(self, genome, config):
        node_tracker = {node_id:{'depth':0, 'output_ids':[], 'input_ids':[]} for node_id in genome.nodes}
        for node_id in config.genome_config.input_keys:
            node_tracker[node_id] = {'depth':0, 'output_ids':[], 'input_ids':[]}
        trace_stack = [node_id for node_id in config.genome_config.input_keys]

        for connection in genome.connections:
            node_tracker[connection[0]]['output_ids'].append(connection[1])
            node_tracker[connection[1]]['input_ids'].append(connection[0])

        while len(trace_stack) > 0:
            trace = trace_stack[0]
            my_depth = node_tracker[trace]['depth']
            next_depth = my_depth + 1
            for output_id in node_tracker[trace]['output_ids']:
                node_tracker[output_id]['depth'] = max(node_tracker[output_id]['depth'], next_depth)
                trace_stack.append(output_id)
            del(trace_stack[0])

        for node_id, node in node_tracker.items():
            node['output_layers']=[]
            node['needs_skip'] = False
            node['id'] = node_id
            for output_id in node['output_ids']:
                node['output_layers'].append(node_tracker[output_id]['depth'])
                if node_tracker[output_id]['depth'] > (node['depth']+1):
                    node['needs_skip'] = True

        for node_id, node in node_tracker.items():
            node['input_layers'] = []
            node['skip_layer_input'] = False
            for input_id in node['input_ids']:
                node['input_layers'].append(node_tracker[input_id]['depth'])
                if node_tracker[input_id]['depth'] < (node['depth']-1):
                    node['skip_layer_input'] = True

        layers = {}
        for node_id, node in node_tracker.items():
            if not node['depth'] in layers:
                layers[node['depth']] = {
                    'nodes':{node_id:node}
                }
            else:
                layers[node['depth']]['nodes'][node_id] = node
                
        # Ensure all nodes have a layer index
        for layer_id, layer in layers.items():
            layer_index = 0
            for node_id, node in layer['nodes'].items():
                node['layer_index'] = layer_index
                layer_index += 1

        
        for layer_id, layer in layers.items():
            layer['is_output_layer'] = False
            layer['is_input_layer'] = False
            layer['layer_type'] = LAYER_TYPE_CONNECTED
            # If I have the output node in me, then I am an output
            if 0 in layer['nodes']:
                layer['is_output_layer'] = True
                layer['layer_type'] = LAYER_TYPE_OUTPUT

            # If I have the first input in me, then I am the input
            if -1 in layer['nodes']:
                layer['is_input_layer'] = True
                layer['layer_type'] = LAYER_TYPE_INPUT

            layer['input_layers'] = []
            ## Compute the shape of required inputs
            for node_id, node in layer['nodes'].items():
                for in_layer in node['input_layers']:
                    if in_layer not in layer['input_layers']:
                        layer['input_layers'].append(in_layer)
            layer['input_layers'].sort()
            layer['input_shape'] = sum(len(layers[jj]['nodes']) for jj in layer['input_layers'])
            layer['weights_shape'] = (layer['input_shape'], len(layer['nodes']))


            # Handle output layer "edge" case
            if layer['is_output_layer']:
                layer['out_weights'] = []
                layer['bias'] = [genome.nodes[node_id].bias for node_id, node in layer['nodes'].items()]
                layer['in_weights'] = [[0 for __ in layers[layer_id-1]['nodes']] for _ in layer['nodes']]
            # Handle input layer "edge" case
            elif layer['is_input_layer']:
                layer['in_weights'] = []
                layer['bias'] = []
                layer['out_weights'] = [[0 for __ in layers[layer_id+1]['nodes']] for _ in layer['nodes']]
            # Handle generic case
            else:
                layer['out_weights'] = [[0 for __ in layers[layer_id+1]['nodes']] for _ in layer['nodes']]
                layer['in_weights'] = [[0 for __ in layers[layer_id-1]['nodes']] for _ in layer['nodes']]

                layer['bias'] = [genome.nodes[node_id].bias for node_id, node in layer['nodes'].items()]
                # else:
                    # layer['bias'] = [0 for _ in layer['nodes']]
            
            layer_index = 0          
            for node_id, node in layer['nodes'].items():
                node['layer_index'] = layer_index
                layer_index += 1


            # Set up current weights
            layer['input_weights'] = np.zeros(layer['weights_shape'])
            layer['input_map'] = {}
            layer_offset = 0
            # Check every layer and every node for connections
            for input_layer_id in layer['input_layers']:
                input_layer = layers[input_layer_id]
                for node_id, node in input_layer['nodes'].items():
                    for node_output_id in node['output_ids']:
                        if node_output_id in layer['nodes']:
                            node_output = layer['nodes'][node_output_id]
                            # I HAVE THIS NODE!
                            # What is it's weight?
                            connection = genome.connections[(node_id, node_output_id)]

                            if not connection.enabled:
                                continue
                            connection_weight = connection.weight

                            in_weight_location = layer_offset + node['layer_index']
                            out_weight_location = node_output['layer_index']
                            layer['input_weights'][in_weight_location][out_weight_location] = connection_weight
                            layer['input_map'][(node_id, node_output_id)] = (in_weight_location, out_weight_location)
                layer_offset += len(input_layer['nodes'])

        return layers

    def update_genome_weights(self):
        for layer_id, layer in self.layers.items():
            for genome_location, weight_location in layer['input_map'].items():
                self.genome.connections[genome_location].new_weight = self.weights[layer_id][weight_location[0]][weight_location[1]].item()
            # layer_offset = 0
            # # Check every layer and every node for connections
            # for input_layer_id in layer['input_layers']:
            #     input_layer = self.layers[input_layer_id]
            #     for node_id, node in input_layer['nodes'].items():
            #         for node_output_id in node['output_ids']:
            #             if node_output_id in layer['nodes']:
            #                 node_output = layer['nodes'][node_output_id]
            #                 # I HAVE THIS NODE!
            #                 print(node_id, node_output_id)
            #                 # What is it's weight?
            #                 connection = self.genome.connections[(node_id, node_output_id)]
            #                 if not connection.enabled:
            #                     continue
            #                 # connection_weight = connection.weight

            #                 in_weight_location = layer_offset + node['layer_index']
            #                 out_weight_location = node_output['layer_index']

            #                 # layer['input_weights'][in_weight_location][out_weight_location] = connection_weight
            #                 self.genome.connections[(node_id, node_output_id)].weight = self.weights[layer_id][in_weight_location][out_weight_location]
                # layer_offset += len(input_layer['nodes'])

    

    @staticmethod
    def _tt(mat):
        return torch.nn.Parameter(torch.tensor(mat,dtype=torch.float64), requires_grad=True)
    def forward(self, x):
        # print("running forward")
        self._outputs = {}
        for layer_id in range(self.n_layers):
            layer_input = None

            layer_weight = None
            layer_type = self.layer_types[layer_id]

            # print(layer_id)
            # print(layer_type)

            if layer_type == LAYER_TYPE_INPUT:
                self._outputs[layer_id] = x
                continue
            ## handle skip layers
            if layer_type == LAYER_TYPE_CONNECTED:
                # print(self.layer_inputs)
                
                # print(self.outputs)
                layer_input = torch.cat([self._outputs[ii] for ii in self.layer_inputs[layer_id]], dim=1)
            ## handle skip layers
            if layer_type == LAYER_TYPE_OUTPUT:
                # print("inputs")
                # print(self.layer_inputs)
                # print("outputs")
                
                # print(self._outputs)

                try:
                    layer_input = torch.cat([self._outputs[ii] for ii in self.layer_inputs[layer_id]], dim=1)
                except:
                    print(layer_type)
                    print(layer_id)
                    print(self._outputs)
                    print("---------------")
                    print(self.layers)
                    layer_input = torch.cat([self._outputs[ii] for ii in self.layer_inputs[layer_id]])

            # print(layer_input)
            # print(self.weights[layer_id])
            # print(self.biases[layer_id])

            self._outputs[layer_id] = torch.sigmoid( torch.matmul(layer_input, self.weights[layer_id]) + self.biases[layer_id] )

            if layer_type == LAYER_TYPE_OUTPUT:
                return self._outputs[layer_id]

    def optimise(self, xs, ys, nEpochs = 100):
    
        USE_CUDA = torch.cuda.is_available()
        # USE_CUDA = False
        device = torch.device("cuda:0" if USE_CUDA else "cpu")
        if not type(xs) is torch.Tensor:
            xs = torch.tensor(xs).to(device)
        if not type(ys) is torch.Tensor:
            ys = torch.tensor(ys).to(device)

        # print('going to train for %s epochs' % nEpochs)
        for epoch in range(nEpochs):
            for inX in range(len(xs)):
                self.optimizer.zero_grad()   # zero the gradient buffers

                output = self.forward(xs[inX])
                target = ys[inX].view(-1)
                target = target.to('cpu')
                loss = self.criterion(output.float(), target.float())
                loss.backward()
                self.optimizer.step()
    optimize = optimise