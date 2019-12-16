#!/data/anaconda3/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pickle as pkl
import scipy.sparse as sp
import sys
import os
import pandas as pd
import networkx as nx
import json
from networkx.readwrite import json_graph



def load_data_radeliu(dataset_route):
    adj = sp.load_npz(dataset_route + "/" + "adj.npz")
    train_index = pd.read_csv(dataset_route + "/" + "train_index.txt", header=None, engine='python').iloc[:,
                  0].values  # .tolist()
    test_index = pd.read_csv(dataset_route + "/" + "test_index.txt", header=None, engine='python').iloc[:,
                 0].values  # .tolist()
    val_index = pd.read_csv(dataset_route + "/" + "val_index.txt", header=None, engine='python').iloc[:,
                0].values  # .tolist()
    train_y = pd.read_csv(dataset_route + "/" + "train_y.txt", header=None, engine='python').iloc[:, 0].values
    val_y = pd.read_csv(dataset_route + "/" + "val_y.txt", header=None, engine='python').iloc[:, 0].values
    feature = sp.csr_matrix(np.load(dataset_route + "/" + "feature.npz")['arr_0']).tolil()
    # feature = sp.identity(adj.shape[0])
    # feature = sp.csr_matrix(np.array([0.5, 0.5] * adj.shape[0]).reshape((adj.shape[0], 2))).tolil()

    return adj, train_index, test_index, val_index, train_y, val_y, feature


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


if __name__ == "__main__":

    file_route = "/data/radeliu/ceph/radeliu/campany_relation/{:s}".format(sys.argv[1])

    adj, train_index, test_index, val_index, train_y, val_y, features = load_data_radeliu(file_route)

    G = nx.from_scipy_sparse_matrix(adj)

    y_train = train_y.reshape((len(train_y), 1))
    y_val = val_y.reshape((len(val_y), 1))
    y_test = np.random.randint(0, 2, len(test_index)).reshape((len(test_index), 1))

    idMap = {}
    classMap = {}
    classmap_train = dict(zip(train_index.astype(str), y_train))
    classmap_val = dict(zip(val_index.astype(str), y_val))
    classmap_test = dict(zip(test_index.astype(str), y_test))

    classMap.update(classmap_train)
    classMap.update(classmap_val)
    classMap.update(classmap_test)

    print("sublength", (len(y_train), len(y_val), len(y_test)))
    print("length:", len(y_train) + len(y_val) + len(y_test))

    for i in range(len(y_train) + len(y_val) + len(y_test)):
        if i in val_index:
            G.node[i]['val'] = True
            G.node[i]['test'] = False
            idMap[i] = i
        elif i in test_index:
            G.node[i]['test'] = True
            G.node[i]['val'] = False
            idMap[i] = i
        else:
            G.node[i]['test'] = False
            G.node[i]['val'] = False
            idMap[i] = i

    data = json_graph.node_link_data(G)

    with open(file_route + "/" + "campany-G.json", "w") as f:
        json.dump(data, f, cls=MyEncoder)

    with open(file_route + "/" + "campany-id_map.json", "w") as f:
        json.dump(idMap, f, cls=MyEncoder)

    with open(file_route + "/" + "campany-class_map.json", "w") as f:
        json.dump(classMap, f, cls=MyEncoder)

    np.save(file_route + "/" + "campany-feats.npy", features.todense())