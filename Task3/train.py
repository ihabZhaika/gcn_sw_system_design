from __future__ import division
from __future__ import print_function

import shutil
import os
import time
import json
# Train on CPU (hide GPU) due to memory constraints
# os.environ['CUDA_VISIBLE_DEVICES'] = ""

import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import networkx as nx
from os import path
from gae.optimizer import OptimizerAE, OptimizerVAE
from gae.model import GCNModelAE, GCNModelVAE
from gae.preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple
from BipartireData.generator import get_train_data_modular, draw_3_sample_modularity_graphs, \
    draw_3_sample_modularity_matrix

placeholders = {
    'features': tf.sparse_placeholder(tf.float32),
    'adj_input': tf.sparse_placeholder(tf.float32),
    'adj_label': tf.sparse_placeholder(tf.float32),
    'dropout': tf.placeholder_with_default(0., shape=())
}


def change_inputs(input_adjs, features):
    new_adjs = []
    ff = []
    for adj_orig in input_adjs:
        adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
        adj_orig.eliminate_zeros()
        new_adjs.append(preprocess_graph(adj_orig))
    ff = [sparse_to_tuple(f.tocoo()) for f in features]
    return new_adjs, ff


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 1000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_string('model', 'gcn_ae', 'Model string.')
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')

model_str = FLAGS.model
dataset_str = FLAGS.dataset


def construct_predicted_graph(new_graph, nodes_count):
    edges_indices = sigmoid(new_graph) >= 0.51
    edges_indices = np.reshape(edges_indices, (nodes_count, nodes_count))
    new_adj = np.zeros((nodes_count, nodes_count))
    new_adj[edges_indices] = 1
    predicted_graph = nx.from_numpy_matrix(new_adj)
    return predicted_graph


def train(nodes_count, graphs_count, test_graphs_count, density, outliers_density, modules_count, max_module_size,
          use_features,
          optimizer_name, save_folder, epochs, config_index):
    enable_global_opt = False
    if graphs_count >= 50:
        enable_global_opt = True
    run_line = " train({nodes_count}, {graphs_count}, {test_graphs_count}, {density}, {outliers_density}, {modules_count}, {max_module_size}, {use_features}, {optimizer_name}, {save_folder}, {epochs}, {config_index})".format(
        nodes_count=nodes_count, graphs_count=graphs_count, test_graphs_count=test_graphs_count,
        density=density, outliers_density=outliers_density, modules_count=modules_count,
        max_module_size=max_module_size, use_features=use_features, optimizer_name=optimizer_name,
        save_folder=save_folder, epochs=epochs, config_index=config_index)
    print(run_line)
    graph_side_nodes = int(nodes_count / 2)
    print("Generating data")
    input_adjs, output_adjs, features = get_train_data_modular(graphs_count, graph_side_nodes, modules_count,
                                                               max_module_size,
                                                               density, outliers_density, use_features)
    test_input_adjs, test_output_adjs, test_features = get_train_data_modular(test_graphs_count, graph_side_nodes,
                                                                              modules_count,
                                                                              max_module_size, density,
                                                                              outliers_density, use_features)
    print("Generating data done, Train shape: {}, Test Shape: {}".format(input_adjs[0].shape, test_input_adjs[0].shape))

    new_input_adjs, features = change_inputs(input_adjs, features)
    test_new_input_adjs, test_features = change_inputs(test_input_adjs, test_features)

    num_features = features[0][2][1]
    features_nonzero = features[0][1].shape[0]

    # Create model
    model = None
    if model_str == 'gcn_ae':
        model = GCNModelAE(placeholders, num_features, features_nonzero)
    elif model_str == 'gcn_vae':
        model = GCNModelVAE(placeholders, num_features, nodes_count, features_nonzero)

    opt = None

    optimizers = []
    tests_optimizers = []
    adjs_sum = 0

    for adj in test_input_adjs:
        adj_sum = adj.sum()
        if adj_sum == 0:
            adj_sum = 1
        pos_weight = float(nodes_count * nodes_count - adj_sum) / adj_sum
        norm = nodes_count * nodes_count / float((nodes_count * nodes_count - adj_sum) * 2)
        if model_str == 'gcn_ae':
            opt = OptimizerAE(preds=model.reconstructions,
                              labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_label'],
                                                                          validate_indices=False), [-1]),
                              pos_weight=pos_weight,
                              norm=norm)
        elif model_str == 'gcn_vae':
            opt = OptimizerVAE(preds=model.reconstructions,
                               labels=tf.reshape(
                                   tf.sparse_tensor_to_dense(placeholders['adj_label'], validate_indices=False),
                                   [-1]),
                               model=model, num_nodes=nodes_count,
                               pos_weight=pos_weight,
                               norm=norm)
        tests_optimizers.append(opt)
    for adj in input_adjs:
        if not enable_global_opt:
            adj_sum = adj.sum()
            if adj_sum == 0:
                adj_sum = 1
            pos_weight = float(nodes_count * nodes_count - adj_sum) / adj_sum
            norm = nodes_count * nodes_count / float((nodes_count * nodes_count - adj_sum) * 2)

            if model_str == 'gcn_ae':
                opt = OptimizerAE(preds=model.reconstructions,
                                  labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_label'],
                                                                              validate_indices=False), [-1]),
                                  pos_weight=pos_weight,
                                  norm=norm)
            elif model_str == 'gcn_vae':
                opt = OptimizerVAE(preds=model.reconstructions,
                                   labels=tf.reshape(
                                       tf.sparse_tensor_to_dense(placeholders['adj_label'], validate_indices=False),
                                       [-1]),
                                   model=model, num_nodes=nodes_count,
                                   pos_weight=pos_weight,
                                   norm=norm)
            optimizers.append(opt)

        adjs_sum += adj.sum()
    adjs_sum /= len(input_adjs)
    if adjs_sum == 0:
        adjs_sum = 1
    pos_weight_all = float(nodes_count * nodes_count - adjs_sum) / adjs_sum
    norm_all = nodes_count * nodes_count / float((nodes_count * nodes_count - adjs_sum) * 2)

    if enable_global_opt:
        # Optimizer
        with tf.name_scope('optimizer'):
            if model_str == 'gcn_ae':
                opt = OptimizerAE(preds=model.reconstructions,
                                  labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_label'],
                                                                              validate_indices=False), [-1]),
                                  pos_weight=pos_weight_all,
                                  norm=norm_all)
            elif model_str == 'gcn_vae':
                opt = OptimizerVAE(preds=model.reconstructions,
                                   labels=tf.reshape(
                                       tf.sparse_tensor_to_dense(placeholders['adj_label'], validate_indices=False),
                                       [-1]),
                                   model=model, num_nodes=nodes_count,
                                   pos_weight=pos_weight_all,
                                   norm=norm_all)
    # Initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    g_avg_accuracy = 0
    # Train model
    train_data = []
    train_folder = path.join(save_folder, "train")
    epoch_folder = path.join(train_folder, str(config_index))
    if not path.exists(train_folder):
        os.mkdir(train_folder)
    if not path.exists(epoch_folder):
        os.mkdir(epoch_folder)
    for epoch in range(epochs):
        t = time.time()
        # Construct feed dictionary
        avg_accuracy = 0
        avg_cost = 0
        index = 0
        result_graphs_paths = []
        for input_adj, label_adj, feature in zip(new_input_adjs, output_adjs, features):
            if not enable_global_opt:
                # Optimizer
                with tf.name_scope('optimizer'):
                    opt = optimizers[index]
            label_adj_n = label_adj + sp.eye(label_adj.shape[0])
            label_adj_n = sparse_to_tuple(label_adj_n)
            feed_dict = construct_feed_dict(input_adj, label_adj_n, feature, placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})
            # Run single weight update
            outs = sess.run([opt.opt_op, opt.cost, opt.accuracy, model.reconstructions],
                            feed_dict=feed_dict)
            # Compute average loss
            avg_cost += outs[1]
            avg_accuracy += outs[2]
            new_graph = outs[3]
            # last round
            if epoch == epochs - 1:
                input_graph = nx.from_numpy_matrix(input_adjs[index].toarray())
                ground_truth_graph = nx.from_numpy_matrix(label_adj.toarray())
                predicted_graph = construct_predicted_graph(new_graph, nodes_count)
                img_path1 = path.join(epoch_folder,
                                      "train_epoch_{epoch}_accuracy_{accuracy}_graph.png".format(epoch=epoch,
                                                                                                 accuracy=avg_accuracy))
                img_path2 = path.join(epoch_folder,
                                      "train_epoch_{epoch}_accuracy_{accuracy}_matrix.png".format(epoch=epoch,
                                                                                                  accuracy=avg_accuracy))
                draw_3_sample_modularity_graphs(input_graph, ground_truth_graph, predicted_graph, save_loc=img_path1,
                                                show=False)
                draw_3_sample_modularity_matrix(input_graph, ground_truth_graph, predicted_graph, save_loc=img_path2,
                                                show=False)
                result_graphs_paths.append(img_path1)
                result_graphs_paths.append(img_path2)

            index += 1

        avg_accuracy /= len(new_input_adjs)
        g_avg_accuracy = avg_accuracy
        avg_cost /= len(new_input_adjs)
        elapsed_time = time.time() - t

        if epoch % 200 == 0 or epoch == epochs - 1:
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
                  "train_acc=", "{:.5f}".format(avg_accuracy),
                  "time=", "{:.5f}".format(elapsed_time))

            train_data.append({
                "epoch": epoch,
                "accuracy": str(avg_accuracy),
                "lost": str(avg_cost),
                "executionTime": elapsed_time,
                "result_graphs": result_graphs_paths
            })
        # if epoch == epochs - 1:
        #     model_path = path.join(epoch_folder, "epoch_{epoch}.ckpt".format(epoch=epoch))
        #     model.save(model_path, sess)

    index = 0
    tests_data = []
    g_avg_test_accuracy = 0
    tests_folder = path.join(save_folder, "tests")
    epoch_folder = path.join(tests_folder, str(config_index))
    if not path.exists(tests_folder):
        os.mkdir(tests_folder)
    if not path.exists(epoch_folder):
        os.mkdir(epoch_folder)
    for input_adj, label_adj, feature in zip(test_new_input_adjs, test_output_adjs, test_features):
        opt = tests_optimizers[index]
        n_label_adj = label_adj + sp.eye(label_adj.shape[0])
        n_label_adj = sparse_to_tuple(n_label_adj)
        feed_dict = construct_feed_dict(input_adj, n_label_adj, feature, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        # Run single weight update
        outs = sess.run([opt.accuracy, model.reconstructions],
                        feed_dict=feed_dict)
        # Compute average loss
        avg_accuracy = outs[0]  # Compute average loss
        print("Test: {} has Acc: {}".format(index, avg_accuracy))
        g_avg_test_accuracy += avg_accuracy
        new_graph = outs[1]
        # roc_curr, ap_curr = ge
        original_graph = nx.from_numpy_matrix(test_output_adjs[index].toarray())
        ground_truth_graph = nx.from_numpy_matrix(label_adj.toarray())
        predicted_graph = construct_predicted_graph(new_graph, nodes_count)
        img_path1 = path.join(epoch_folder,
                              "test_index_{index}_accuracy_{accuracy}_graph.png".format(index=index,
                                                                                        accuracy=avg_accuracy))
        img_path2 = path.join(epoch_folder,
                              "test_index_{index}_accuracy_{accuracy}_matrix.png".format(index=index,
                                                                                         accuracy=avg_accuracy))
        draw_3_sample_modularity_graphs(original_graph, ground_truth_graph, predicted_graph, save_loc=img_path1,
                                        show=False)
        draw_3_sample_modularity_matrix(original_graph, ground_truth_graph, predicted_graph, save_loc=img_path2,
                                        show=False)
        tests_data.append({
            "index": index,
            "accuracy": str(avg_accuracy),
            "result_graph": img_path1,
            "result_matrix": img_path2
        })
        index += 1
    g_avg_test_accuracy /= test_graphs_count
    print("train finished")
    results = {
        "configurationIndex": config_index,
        "modelName": optimizer_name,
        "epochs": epochs,
        "nodes_count": nodes_count, "graphs_count": graphs_count, "test_graphs_count": test_graphs_count,
        "density": str(density), "outliers_density": str(outliers_density),
        "use_features": use_features, "optimizer_name": optimizer_name,
        "train_folder": path.join(save_folder, "train", str(config_index)),
        "test_folder": path.join(save_folder, "test", str(config_index)),
        "train_accuracy": str(g_avg_accuracy),
        "test_accuracy": str(g_avg_test_accuracy),
        "modules_count": modules_count,
        "module_max_size": module_max_size,
        "tests_data": tests_data,
        "train_data": train_data,
        "run_line": run_line
    }
    return results


if __name__ == "__main__":

    # train(8, 1, 1, 0.6, 0.1, 6, 6, False, "gcn_ae", "C:\\Users\Ihab\PycharmProjects\gae\\run_data_task_3\\1587335477",
    #       1200, 14)

    config_index = 0
    maximum_config_index = 0
    epochs = 1200
    max_nodes = 44
    max_graphs_count = 100
    max_density = 80
    # max_false_edge_density = 40
    modules_max_count = 6
    module_max_size = 6
    use_features = False
    optimizer_name = FLAGS.model
    storage_path = "C:\\Users\Ihab\PycharmProjects\gae\\run_data_task_3"
    save_folder = path.join(str(int(time.time())))
    stamp = str(int(time.time()))
    run_path = path.join(storage_path, stamp)
    os.mkdir(run_path)
    print("Data Path: {}".format(save_folder))
    results = []
    best_configurations = []

    density = 50
    graphs_count = 25
    nodes_count = 40
    modules_count = 6

    for module_size in range(1, module_max_size + 1, 1):
        for false_density in range(10, 51, 10):
            maximum_test_accuracy = -1
            config_index += 1
            density_frac = density / 100
            false_density_frac = false_density / 100
            test_graphs_count = int((graphs_count * 0.3)) + 1
            epochs = epochs + (graphs_count * nodes_count)
            if epochs > 1200:
                epochs = 1200
            res = train(nodes_count, graphs_count, test_graphs_count, density_frac, false_density_frac,
                        modules_count, module_size,
                        use_features, optimizer_name, run_path, epochs, config_index)

            test_accuracy = float(res["test_accuracy"])
            train_accuracy = float(res["train_accuracy"])
            run_line = res["run_line"]
            if test_accuracy > maximum_test_accuracy:
                maximum_test_accuracy = test_accuracy
                maximum_config_index = config_index
                configs_path = path.join(run_path, "bestConfigs")
                if not path.exists(configs_path):
                    os.mkdir(configs_path)
                with open(path.join(configs_path,
                                    "best_config_{}_graphs_{}_nodes_{}.json".format(config_index,
                                                                                    graphs_count,
                                                                                    nodes_count)),
                          "w") as f:
                    conf = {
                        "configurationIndex": maximum_config_index,
                        "nodesCount": nodes_count,
                        "graphsCount": graphs_count,
                        "testGraphsCount": test_graphs_count,
                        "trainAccuracy": train_accuracy,
                        "testAccuracy": test_accuracy,
                        "density": density_frac,
                        "false_density": false_density_frac,
                        "useFeatures": use_features,
                        "runLine": run_line,
                        "model": optimizer_name,
                        "modules_count": modules_count,
                        "module_max_size": module_max_size

                    }
                    best_configurations.append(conf)
                    data_txt = json.dumps(conf, indent=4)
                    f.write(data_txt)

                    final_res = {
                        "stamp": stamp,
                        "path": run_path,
                        "results": results,
                        "bestConfigurations": best_configurations
                    }
                    print("<-------New Best configuration------->\n{}".format(data_txt))
            else:
                # we will delete this configuration files
                train_epoch_folder = path.join(run_path, "train", str(config_index))
                tests_epoch_folder = path.join(run_path, "tests", str(config_index))
                shutil.rmtree(train_epoch_folder)
                shutil.rmtree(tests_epoch_folder)
            results.append(res)
            with open(path.join(run_path, "results.json"), "w") as f:
                json.dump(final_res, f, indent=4)
                print("Result saved")
    # for graphs_count in range(50, max_graphs_count + 1, 25):
    #     for nodes_count in range(32, max_nodes + 1, 8):
    #         maximum_test_accuracy = 0
    #         for modules_count in range(2, modules_max_count + 1, 2):
    #             if modules_count > nodes_count / 2:
    #                 continue
    #             for density in range(50, max_density + 1, 20):
    #                 for false_density in range(10, density + 1, 10):
    #                     config_index += 1
    #                     density_frac = density / 100
    #                     false_density_frac = false_density / 100
    #                     test_graphs_count = int((graphs_count * 0.3)) + 1
    #                     epochs = epochs + (graphs_count * nodes_count)
    #                     if epochs > 1200:
    #                         epochs = 1200
    #                     res = train(nodes_count, graphs_count, test_graphs_count, density_frac, false_density_frac,
    #                                 modules_count, module_max_size,
    #                                 use_features, optimizer_name, run_path, epochs, config_index)
    #
    #                     test_accuracy = float(res["test_accuracy"])
    #                     train_accuracy = float(res["train_accuracy"])
    #                     run_line = res["run_line"]
    #                     if test_accuracy > maximum_test_accuracy:
    #                         maximum_test_accuracy = test_accuracy
    #                         maximum_config_index = config_index
    #                         configs_path = path.join(run_path, "bestConfigs")
    #                         if not path.exists(configs_path):
    #                             os.mkdir(configs_path)
    #                         with open(path.join(configs_path,
    #                                             "best_config_{}_graphs_{}_nodes_{}.json".format(config_index,
    #                                                                                             graphs_count,
    #                                                                                             nodes_count)),
    #                                   "w") as f:
    #                             conf = {
    #                                 "configurationIndex": maximum_config_index,
    #                                 "nodesCount": nodes_count,
    #                                 "graphsCount": graphs_count,
    #                                 "testGraphsCount": test_graphs_count,
    #                                 "trainAccuracy": train_accuracy,
    #                                 "testAccuracy": test_accuracy,
    #                                 "density": density_frac,
    #                                 "false_density": false_density_frac,
    #                                 "useFeatures": use_features,
    #                                 "runLine": run_line,
    #                                 "model": optimizer_name,
    #                                 "modules_count": modules_count,
    #                                 "module_max_size": module_max_size
    #
    #                             }
    #                             best_configurations.append(conf)
    #                             data_txt = json.dumps(conf, indent=4)
    #                             f.write(data_txt)
    #
    #                             final_res = {
    #                                 "stamp": stamp,
    #                                 "path": run_path,
    #                                 "results": results,
    #                                 "bestConfigurations": best_configurations
    #                             }
    #                             print("<-------New Best configuration------->\n{}".format(data_txt))
    #                     else:
    #                         # we will delete this configuration files
    #                         train_epoch_folder = path.join(run_path, "train", str(config_index))
    #                         tests_epoch_folder = path.join(run_path, "tests", str(config_index))
    #                         shutil.rmtree(train_epoch_folder)
    #                         shutil.rmtree(tests_epoch_folder)
    #                     results.append(res)
    #                     with open(path.join(run_path, "results.json"), "w") as f:
    #                         json.dump(final_res, f, indent=4)
    #                         print("Result saved")
