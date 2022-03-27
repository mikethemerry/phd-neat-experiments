def binary_cross_entropy(genomes, config):
    loss = nn.BCELoss()
    loss = loss.to(device)
    for genome_id, genome in genomes.items():
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        preds = []
        for xi in xs:
            preds.append(net.activate(xi))
        genome.fitness = float(
            1./loss(torch.tensor(preds).to(device), torch.tensor(ys)))
