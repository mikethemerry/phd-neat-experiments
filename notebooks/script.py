from collections import defaultdict
import math
import sys
import json


def layers(data):
    adj = defaultdict(dict)
    layers = defaultdict(dict)
    # data = [[4, 0], [1, 0], [2, 0], [3, 0], [3, 4], [0, 6], [2, 4], [0, 5], [5, 6]]
    # I will assume the number for nodes are in some logical way:
    # node number increases from top to down, left to right
    # otherwise I would sort based on a 'topological sort'
    # then create a hashmap for each node and a token from 1 to |V|
    data = sorted(data, key=lambda data: [data[0], data[1]])
    for s, e in data:
        adj[s][e] = e
    ls, le = zip(*data)
    v = {key: -math.inf for key in set([i for li in data for i in li])}

    # modified dfs, updates depth to max(depth_from_dfs, oringial)
    def dfs(node, graph):
        queue = list(graph[node].keys())
        v[node] = 0
        visited = set()
        depth = 0
        while queue:
            this = queue.pop(0)
            if this not in visited:
                depth += 1
                v[this] = max(v[this], depth)
                visited.add(this)
                queue = queue + list(graph[this].keys())

    def getto(d):
        to_d = []
        for i in layers:
            edges = [[s, e] for s in layers[d]['nodes'] for e in layers[i]['nodes']]
            if any([edge in data for edge in edges]):
                to_d.append(i)
        return to_d

    def getfrom(d):
        from_d = []
        for i in layers:
            edges = [[e, s] for s in layers[d]['nodes'] for e in layers[i]['nodes']]
            if any([edge in data for edge in edges]):
                from_d.append(i)
        return from_d

    # dfs for all nodes in layer 0
    for node in set(ls).difference(le):
        dfs(node, adj)

    for d in range(max(list(v.values())) + 1):
        layers[d]['nodes'] = [i for i in v if v[i] == d]
        layers[d]['shallowest?'] = 'True' if d == 0 else ''
        layers[d]['deepest?'] = 'True' if d == max(list(v.values())) else ''

    for d in layers:
        layers[d]['connect_to'] = getto(d)
        layers[d]['connect_from'] = getfrom(d)

    return layers



def process(data):
    data = layers(data)
    result = {
        "processed": True,
        "data": data
    }
    return result


def main(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    result = process(data)
    print(result)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Error: Please specify the path to the input file.')
    else:
        file_path = sys.argv[1]
        main(file_path)
