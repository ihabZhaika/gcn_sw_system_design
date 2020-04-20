from dgl.data import MiniGCDataset
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import bipartite
import random
from dgl.nn.pytorch import GraphConv
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import pylab
from Task1.tsne import tsne
import numpy as np
import json

device = torch.device('cuda:0')


# Similar code can be found on
# https://docs.dgl.ai/tutorials/basics/4_batch.html
# without the customization for bipartite graphs


def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)


class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Classifier, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim).to(device)
        self.conv2 = GraphConv(hidden_dim, hidden_dim).to(device)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g):
        # Use node degree as the initial node feature. For undirected graphs, the in-degree
        # is the same as the out_degree.
        h = g.in_degrees().view(-1, 1).float().to(device)
        # Perform graph convolution and activation function.
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        g.ndata['h'] = h
        # Calculate graph representation by averaging all the node representations.
        hg = dgl.mean_nodes(g, 'h')
        return self.classify(hg)


def get_random_graph(node_count):
    funcs = [nx.complete_graph, nx.circular_ladder_graph,
             nx.path_graph, nx.star_graph, nx.wheel_graph]
    rand = int(random.random() * len(funcs))
    return funcs[rand](node_count)


def get_graphs(node_count, graphs_count):
    graphs = []
    for i in range(graphs_count):
        g = dgl.DGLGraph(get_random_graph(node_count))
        g = g.to(device)
        graphs.append((g, 1))
    for i in range(graphs_count):
        g = dgl.DGLGraph(bipartite.random_graph(node_count, node_count, random.random()))
        g = g.to(device)
        graphs.append((g, 0))
    return graphs


def draw_bipartite_g(ax, g, label):
    ax.set_title(label)
    nodes_count = int(len(g) / 2)
    pos = []
    colors = ["orange"] * nodes_count + ['green'] * nodes_count
    pos.extend([(0, i) for i in range(nodes_count)])  # put nodes from X at x=1
    pos.extend([(1, i) for i in range(nodes_count)])  # put nodes from X at x=1
    nx.draw(g, pos=pos, node_color=colors, ax=ax)


def draw_sample(non_bipartite_g, bipartite_g):
    fig, axes = plt.subplots(nrows=1, ncols=2)
    axes.flat[0].set_title("Non-bipartite")
    nx.draw(non_bipartite_g.to_networkx(), ax=axes.flat[0])
    draw_bipartite_g(axes.flat[1], bipartite_g.to_networkx(), "Bipartite")
    # ax.set_title('Class: {:d}'.format(label))
    plt.show()


max_nodes_count = 50
nodes_count_arr = []
test_accuracy_arr = []
# Create training and test sets.
nodes_count = 15
graph_count = 350

for nodes_count in range(5, max_nodes_count + 1, 5):
    print("Nodes: {}".format(nodes_count))
    trainset = get_graphs(nodes_count, graph_count)
    testset = get_graphs(nodes_count, 50)

    draw_sample(trainset[0][0], trainset[-1][0])
    # Use PyTorch's DataLoader and the collate function
    # defined before.
    data_loader = DataLoader(trainset, batch_size=61, shuffle=True,
                             collate_fn=collate)

    # Create model
    classes_count = 2
    model = Classifier(1, 384, classes_count)
    model = model.to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()

    epoch_losses = []
    for epoch in range(1000):
        epoch_loss = 0
        for iter, (bg, label) in enumerate(data_loader):
            prediction = model(bg)
            label = label.to(device)
            loss = loss_func(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
        epoch_loss /= (iter + 1)
        print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
        epoch_losses.append(epoch_loss)

    plt.title('cross entropy averaged over minibatches')
    plt.plot(epoch_losses)
    plt.show()

    ###############################################################################
    # The trained model is evaluated on the test set created. To deploy
    # the tutorial, restrict the running time to get a higher
    # accuracy (:math:`80` % ~ :math:`90` %) than the ones printed below.

    model.eval()
    # Convert a list of tuples to two lists
    test_X, test_labels = map(list, zip(*testset))
    test_bg = dgl.batch(test_X)
    test_Y = torch.tensor(test_labels).float().view(-1, 1).to(device)
    coordinates = model(test_bg).cpu().detach().numpy()
    probs_Y = torch.softmax(model(test_bg), 1)
    sampled_Y = torch.multinomial(probs_Y, 1)
    argmax_Y = torch.max(probs_Y, 1)[1].view(-1, 1)
    acc1 = (test_Y == sampled_Y.float()).sum().item() / len(test_Y) * 100
    print('Accuracy of sampled predictions on the test set: {:.4f}%'.format(acc1))
    acc2 = (test_Y == argmax_Y.float()).sum().item() / len(test_Y) * 100
    print('Accuracy of argmax predictions on the test set: {:4f}%'.format(acc2))

    Y = tsne(coordinates, 2, 2)
    test_labels = np.array(test_labels)
    pylab.scatter(Y[:, 0], Y[:, 1], 20, test_labels)
    pylab.title("Graph scatter from the latent vector")
    pylab.show()
    pylab.close()
    nodes_count_arr.append(nodes_count)
    test_accuracy_arr.append((acc1, acc2))

    with open("./results.json", "w") as f:
        res = {
            "nodes_count_arr": nodes_count_arr,
            "test_accuracy_arr": test_accuracy_arr
        }
        json.dump(res, f, indent=4)
