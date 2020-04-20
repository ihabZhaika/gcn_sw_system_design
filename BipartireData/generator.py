# import dgl
import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
from networkx.algorithms import bipartite
import scipy.sparse as sp
import math


def add_edges_to_one_side(g, nodes_count, min_node_index, max_node_index, fill_ratio):
    nodes_count_to_link = int(fill_ratio * nodes_count)
    part_a = [random.randint(min_node_index, max_node_index) for i in range(nodes_count_to_link)]
    nodes_count_to_link = int(random.random() * nodes_count)
    part_b = [random.randint(min_node_index, max_node_index) for i in range(nodes_count_to_link)]
    # now connect all of those together
    for node_x in part_a:
        for node_y in part_b:
            g.add_edge(node_x, node_y)


def generate_dataset(graphs_count, min_nodes, max_nodes, link_fill_percentage=random.random(),
                     fake_edges_ratio=random.random(), ):
    labels_graphs = []
    input_graphs = []
    # generate bipartite graphs to be the label
    for i in range(graphs_count):
        nodes_count = random.randrange(min_nodes, max_nodes)
        l_g = bipartite.random_graph(nodes_count, nodes_count, link_fill_percentage)
        labels_graphs.append(l_g)
        # now we will add false edges
        i_g = l_g.copy()
        add_edges_to_one_side(i_g, nodes_count, 0, nodes_count, fake_edges_ratio)
        add_edges_to_one_side(i_g, nodes_count, nodes_count, (nodes_count * 2) - 1, fake_edges_ratio)
        input_graphs.append(i_g)
    return input_graphs, labels_graphs


def get_modules_split(nodes_count, modules_count, max_module_size):

    factorized_numbers = []
    # let's generate k random numbers where their sum is k
    while True:
        numbers = []
        for i in range(modules_count):
            numbers.append(random.random() * max_module_size)
        # after having all of those numbers we want  K1/Z + K2/Z ... = N
        factor = sum(numbers) / nodes_count
        factorized_numbers = [int(num / factor) for num in numbers]
        if sum(factorized_numbers) != 0:
            break
    modules_count = min(max_module_size, nodes_count)
    new_factorized = [
        int(math.fabs(normalize(num, 1, modules_count, min(factorized_numbers), max(factorized_numbers)))) for num in
        factorized_numbers]
    val, min_idx = min((val, idx) for (idx, val) in enumerate(new_factorized))
    val, max_idx = max((val, idx) for (idx, val) in enumerate(new_factorized))
    compensate = nodes_count - sum(new_factorized)
    if compensate > 0:
        new_factorized[min_idx] += compensate
    else:
        new_factorized[max_idx] += compensate

    # some of them will have fractions, so take the int part and the left amount add it to the first one
    assert len([num for num in new_factorized if num < 0]) == 0
    assert sum(new_factorized) == nodes_count
    return new_factorized


def generate_modular_graph_labels(module_size, start_number):
    nodes_labels = {}
    nodes_labels.update({i: "S{}".format(i + start_number) for i in range(module_size)})
    nodes_labels.update({i + module_size: "F{}".format(i + start_number) for i in range(module_size)})
    return nodes_labels


def get_graph_with_outliers_from_g(g, sub_graphs, modules_nodes, outliers_sparsity):
    new_g = g.copy()
    for i in range(len(sub_graphs) - 1):
        sub_a = sub_graphs[i]
        sub_b = sub_graphs[i + 1]
        edges_count = min(len(sub_a.edges), len(sub_b.edges)) + 1
        outliers_count = int(edges_count * outliers_sparsity)
        # now we want to add those edges between random S from sub1 to F from sub2 and vise versa
        for edge_num in range(outliers_count):
            a = None
            b = None
            if edge_num % 2 == 0:
                a = np.random.choice(modules_nodes[i]["s"])
                b = np.random.choice(modules_nodes[i + 1]["f"])
            else:
                a = np.random.choice(modules_nodes[i + 1]["s"])
                b = np.random.choice(modules_nodes[i]["f"])
            new_g.add_edge(a, b, outlier=1)

    return new_g


def get_modularity_matrix_for_drawing(graph):
    mat = nx.adj_matrix(graph).toarray().astype(np.float32)
    for node_id, neighbour_nodes in graph.adj.items():
        for neighbour_id, label in neighbour_nodes.items():
            if "outlier" in label:
                if label["outlier"] == 0:
                    mat[node_id, neighbour_id] = 1
                else:
                    mat[node_id, neighbour_id] = 0.5
            else:
                # keep the original value
                pass
    mat_x_size = mat.shape[0]
    mat_new = mat[int(mat_x_size / 2):, 0:int(mat_x_size / 2)]
    return mat_new


def generate_dataset_modular(graphs_count, nodes_count, modules_count, max_module_size,
                             module_sparsity=random.random(), outliers_sparsity=random.random()):
    assert modules_count > 0, "Can't have 0 modules"
    assert nodes_count > 0, "Can't have 0 modules"
    original_graphs = []
    graphs_with_outliers = []
    # we want to split into modules_count bipartite graphs, then we will join them together
    for i in range(graphs_count):
        # fig, axs = plt.subplots(nrows=1, ncols=modules_count + 2)
        modules_split = []

        while True:
            # sometimes it throws exceptions on  bad grilled interval, so retry
            try:
                modules_split = get_modules_split(nodes_count, modules_count, max_module_size)
                break
            except Exception as e:
                print("get_modules_split failed, retry")
                continue
        # adjacency_arr = np.zeros((nodes_count * 2, nodes_count * 2))
        total_nodes = 0
        mixed_g = nx.empty_graph(nodes_count * 2)
        idx = 0
        modules_nodes = {}
        sub_graphs = []
        for module_size in modules_split:
            sub_g = bipartite.random_graph(module_size, module_size, module_sparsity)
            sub_graphs.append(sub_g)
            s = []
            f = []
            for edge in sub_g.edges:
                a, b = edge
                a = a + total_nodes
                b = (b - module_size) + nodes_count + total_nodes
                s.append(a)
                f.append(b)
                mixed_g.add_edge(a, b, outlier=0)
            modules_nodes[idx] = {"f": f, "s": s}
            # nodes_labels = generate_modular_graph_labels(module_size, total_nodes)
            # draw_g_modular(axs.flat[idx], sub_g, "G: {}".format(idx), nodes_labels)
            total_nodes += module_size
            idx += 1
        #  now we will create new graph
        g_with_outliers = get_graph_with_outliers_from_g(mixed_g, sub_graphs, modules_nodes, outliers_sparsity)

        original_graphs.append(mixed_g)
        graphs_with_outliers.append(g_with_outliers)

        # draw_g_modular(axs.flat[-2], mixed_g, "Combined", nodes_labels=generate_modular_graph_labels(nodes_count, 0))
        # draw_g_modular(axs.flat[-1], g_with_outliers, "With outliers",
        #                nodes_labels=generate_modular_graph_labels(nodes_count, 0))
        # plt.show()
        # plt.close()
    return original_graphs, graphs_with_outliers


def normalize(x, a, b, min, max):
    return a + (((x - min) / (max - min)) * (b - a))


def draw_g_modular(ax, g, title, nodes_labels=None):
    ax.set_title(title)
    nodes_count = int(len(g) / 2)
    colors = ["orange"] * nodes_count + ['green'] * nodes_count
    pos = []
    pos.extend([(0, i) for i in range(nodes_count)])  # put nodes from X at x=1
    pos.extend([(1, i) for i in range(nodes_count)])  # put nodes from X at x=1
    if nodes_labels is None:
        nx.draw_networkx(g, pos=pos, node_color=colors, ax=ax, )
    else:
        nx.draw_networkx(g, pos=pos, node_color=colors, ax=ax, labels=nodes_labels)


def draw_g(ax, g, title, is_biparted):
    ax.set_title(title)
    nodes_count = int(len(g) / 2)
    colors = ["orange"] * nodes_count + ['green'] * nodes_count
    if is_biparted:
        pos = []
        pos.extend([(0, i) for i in range(nodes_count)])  # put nodes from X at x=1
        pos.extend([(1, i) for i in range(nodes_count)])  # put nodes from X at x=1
        nx.draw(g, pos=pos, node_color=colors, ax=ax)

    else:
        poss = nx.drawing.spring_layout(g)
        new_pos = []
        x_cords = [pos[0] for pos in poss.values()]
        y_cords = [pos[1] for pos in poss.values()]
        for pos in poss.values():
            x = normalize(pos[0], 0, 1, min(x_cords), max(x_cords))
            y = normalize(pos[1], 0, nodes_count, min(y_cords), max(y_cords))
            new_pos.append((x, y))
        nx.draw(g, node_color=colors, ax=ax, pos=new_pos)


def draw_sample(input_g, labels_g, title=""):
    fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, sharex=True)
    fig.suptitle(title, fontsize=16)
    axes_flat = axes.flat
    draw_g(axes_flat[0], labels_g, "Bipartite Graph + False edges", False)
    draw_g(axes_flat[1], input_g, "Bipartite Graph", True)
    plt.show()


def draw_3_sample(original, label, predicted, title="", save_loc=None, show=True):
    fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True, sharex=True)
    fig.suptitle(title, fontsize=16)
    axes_flat = axes.flat
    draw_g(axes_flat[1], label, "Ground Truth Graph", True)
    draw_g(axes_flat[0], original, "Corrupted Graph", False)
    draw_g(axes_flat[2], predicted, "Predicted Graph", True)
    if save_loc:
        plt.savefig(save_loc, dpi=300)
    if show:
        plt.show()
    plt.close(fig)


def get_train_data(graphs_count, min_nodes, max_nodes, link_fill_percentage=random.random(),
                   fake_edges_ratio=random.random(), use_features=False):
    input_graphs, labels_graphs = generate_dataset(graphs_count, min_nodes, max_nodes, link_fill_percentage,
                                                   fake_edges_ratio)
    features = []
    if use_features:
        for g in labels_graphs:
            feature = np.zeros((len(g), 2))
            half_nodes_count = int(len(g) / 2)
            feature[0:half_nodes_count, 0] = 1
            feature[half_nodes_count:, 1] = 1
            features.append(sp.coo_matrix(feature))
    else:
        features = [nx.adjacency_matrix(g) for g in labels_graphs]

    input_graphs = [nx.adjacency_matrix(g) for g in input_graphs]
    labels_graphs = [nx.adjacency_matrix(g) for g in labels_graphs]
    return input_graphs, labels_graphs, features


def get_train_data_modular(graphs_count, nodes_count, modules_count, max_module_size,
                           density=random.random(), outliers_density=random.random(), use_features=False):
    input_graphs, labels_graphs = generate_dataset_modular(graphs_count, nodes_count, modules_count, max_module_size,
                                                           density, outliers_density)
    features = []
    if use_features:
        for g in labels_graphs:
            feature = np.zeros((len(g), 2))
            half_nodes_count = int(len(g) / 2)
            feature[0:half_nodes_count, 0] = 1
            feature[half_nodes_count:, 1] = 1
            features.append(sp.coo_matrix(feature))
    else:
        features = [nx.adjacency_matrix(g) for g in labels_graphs]

    input_graphs = [nx.adjacency_matrix(g) for g in input_graphs]
    labels_graphs = [nx.adjacency_matrix(g) for g in labels_graphs]
    return input_graphs, labels_graphs, features


def draw_modularity_matrix_from_graph(ax, g):
    mat = nx.adj_matrix(g).toarray()
    mat_x_size = mat.shape[0]
    mat_new = mat[int(mat_x_size / 2):, 0:int(mat_x_size / 2)]
    mat_new = mat_new.astype(np.float32)
    draw_modularity_matrix(ax, mat_new, "Title")


def draw_modularity_matrix(ax, matrix, title):
    ax.set_title(title)
    im = ax.imshow(matrix, cmap=plt.cm.Oranges)

    # We want to show all ticks...
    ax.set_xticks(np.arange(matrix.shape[0]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    # ... and label them with the respective list entries
    x_tics = ["S{}".format(i) for i in range(matrix.shape[0])]
    y_tics = ["F{}".format(i) for i in range(matrix.shape[0])]
    ax.xaxis.set_ticks_position('top')
    ax.set_xticklabels(x_tics)
    ax.set_yticklabels(y_tics)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(x_tics)):
        for j in range(len(y_tics)):
            val = "0"
            if matrix[j, i] > 0:
                val = "1"
            text = ax.text(i, j, val, ha="center", va="center", color="black")


def draw_3_sample_modularity_graphs(original, label, predicted, title="", save_loc=None, show=True):
    fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True, sharex=True)
    fig.suptitle(title, fontsize=16)
    axes_flat = axes.flat
    labels = generate_modular_graph_labels(int(len(label) / 2), 0)
    draw_g_modular(axes_flat[1], label, "Modularity graph", labels)
    draw_g(axes_flat[0], original, "Modularity graph with outliers", labels)
    draw_g(axes_flat[2], predicted, "Predicted Graph", labels)
    if save_loc:
        plt.savefig(save_loc, dpi=300)
    if show:
        plt.show()
    plt.close(fig)


def draw_3_sample_modularity_matrix(original, label, predicted, title="", save_loc=None, show=True):
    fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True, sharex=True)
    fig.suptitle(title, fontsize=16)
    axes_flat = axes.flat
    # labels = generate_modular_graph_labels(len(label) / 2, 0)

    draw_modularity_matrix(axes.flat[1], get_modularity_matrix_for_drawing(label), "Modularity matrix")
    draw_modularity_matrix(axes.flat[0], get_modularity_matrix_for_drawing(original),
                           "Modularity matrix\nwith outliers")
    draw_modularity_matrix(axes.flat[2], get_modularity_matrix_for_drawing(predicted), "Predicted \nModularity matrix")

    if save_loc:
        plt.savefig(save_loc, dpi=300)
    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    input_g, labels_g = generate_dataset(10, 5, 10)
    draw_sample(labels_g[5], input_g[5], "")
