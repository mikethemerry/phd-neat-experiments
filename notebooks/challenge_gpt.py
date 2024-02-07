import json
from collections import defaultdict

def build_layers(connections):
    layers = defaultdict(lambda: {"inputs": [], "outputs": []})

    for input_node, output_node in connections:
        input_depth = layers[input_node]["depth"] if "depth" in layers[input_node] else -1
        output_depth = layers[output_node]["depth"] if "depth" in layers[output_node] else -1

        # set the depth of the output node if it hasn't been set already
        if output_depth == -1:
            output_depth = input_depth + 1
            layers[output_node]["depth"] = output_depth

        # update the input node's output connections
        layers[input_node]["outputs"].append(output_depth)

        # update the output node's input connections
        layers[output_node]["inputs"].append(input_depth)

    # group nodes by depth
    layers_by_depth = defaultdict(list)
    for node, node_data in layers.items():
        if "depth" in node_data:
            layers_by_depth[node_data["depth"]].append(node)

    # compute properties of each layer
    for depth, nodes in layers_by_depth.items():
        layer = layers[ nodes[0] ]
        layer["is_shallowest_layer"] = depth == 0
        layer["is_deepest_layer"] = len(layers_by_depth) - 1 == depth
        layer["input_layers"] = sorted(set([ layers[input_node]["depth"] for node in nodes for input_node in layers[node]["inputs"] ]))
        layer["output_layers"] = sorted(set([ layers[output_node]["depth"] for node in nodes for output_node in layers[node]["outputs"] ]))
        layer["nodes"] = nodes

    # sort layers by depth
    layers = [layers[ nodes[0] ] for depth, nodes in sorted(layers_by_depth.items())]

    return layers

if __name__ == "__main__":
    import sys

    # load input data from a file
    input_file_path = sys.argv[1]
    with open(input_file_path, "r") as input_file:
        input_data = json.load(input_file)

    # extract the connections from the input data
    connections = input_data

    # build the layers data structure
    layers = build_layers(connections)

    # print the layers data structure
    print(json.dumps(layers, indent=4))
