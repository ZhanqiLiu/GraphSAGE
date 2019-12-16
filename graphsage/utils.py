from __future__ import print_function

import tensorflow as tf
import numpy as np
import random
import json
import sys
import os
from sklearn.preprocessing import StandardScaler
#predict
import time
import scipy.sparse as sp

from graphsage.partition_utils import partition_graph

import networkx as nx
from networkx.readwrite import json_graph
version_info = list(map(int, nx.__version__.split('.')))
major = version_info[0]
minor = version_info[1]
#assert (major <= 1) and (minor <= 11), "networkx major version > 1.11"


def load_data(prefix, normalize=True, load_walks=False):
    G_data = json.load(open(prefix + "-G.json"))
    G = json_graph.node_link_graph(G_data)
    if isinstance(G.nodes()[0], int):
        conversion = lambda n : int(n)
    else:
        conversion = lambda n : n

    if os.path.exists(prefix + "-feats.npy"):
        feats = np.load(prefix + "-feats.npy")
    else:
        print("No features present.. Only identity features will be used.")
        feats = None

    id_map = json.load(open(prefix + "-id_map.json"))

    print("id_map load keys:", list(id_map.keys())[:1000])

    id_map = {int(k): int(v) for k,v in id_map.items()}

    print("id_map after process keys:", list(id_map.keys())[:1000])

    print("id_map length:", len(id_map.keys()))
    print("450 id map:", id_map[450])

    walks = []
    class_map = json.load(open(prefix + "-class_map.json"))
    if isinstance(list(class_map.values())[0], list):
        lab_conversion = lambda n : n
    else:
        lab_conversion = lambda n : int(n)


    print("class_map load keys:", list(class_map.keys())[:1000])

    class_map = {int(k): lab_conversion(v) for k,v in class_map.items()}

    print("class_map after process keys:", list(class_map.keys())[:1000])


    ## Remove all nodes that do not have val/test annotations(注释)
    ## (necessary because of networkx weirdness with the Reddit data)
    broken_count = 0
    for node in G.nodes():
        if not 'val' in G.node[node] or not 'test' in G.node[node]:
            G.remove_node(node)
            broken_count += 1
    print("Removed {:d} nodes that lacked proper annotations due to networkx versioning issues".format(broken_count))

    ## Make sure the graph has edge train_removed annotations
    ## (some datasets might already have this..)
    print("Loaded data.. now preprocessing..")
    #如果一条边的两个点，有一个是val和test，那么这条边要被标记为train_removed
    for edge in G.edges():
        if (G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or
            G.node[edge[0]]['test'] or G.node[edge[1]]['test']):
            #G[edge[0]][edge[1]]['train_removed'] = True
            G[edge[0]][edge[1]]['train_removed'] = False#radeliu 修改
        else:
            G[edge[0]][edge[1]]['train_removed'] = False
#feats is not None

    print("length1:", (len(G.nodes()), len(id_map), len(class_map), feats.shape))

    if normalize and not feats is None:
        from sklearn.preprocessing import StandardScaler
        #train_ids = np.array([id_map[n] for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']])
        train_ids=[]
        for n in G.nodes():
            #print(n)
            if not G.node[n]['val'] and not G.node[n]['test']:
                #print("past G")
                #n = str(n)
                train_ids.append(id_map[n])
                #print("past id_map")
        train_ids = np.array(train_ids)
        train_feats = feats[train_ids]
        scaler = StandardScaler()
        scaler.fit(train_feats)
        feats = scaler.transform(feats)

    return G, feats, id_map, class_map


def uniform(shape, scale=0.05, name=None):
    """Uniform init."""
    initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def ones(shape, name=None):
    """All ones."""
    initial = tf.ones(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def load_predict_data(dataset_path, normalize=True):
  """Load GraphSAGE data."""
  start_time = time.time()

  adj = sp.load_npz(dataset_path + "adj_all.npz")
  feats = np.load(dataset_path + "feature.npz")
  feats = feats["arr_0"]
  id_map = json.load(open(dataset_path + "campany-id_map.json"))

  is_digit = list(id_map.keys())[0].isdigit()
  id_map = {(int(k) if is_digit else k): int(v) for k, v in id_map.items()}

  num_data = len(id_map)

  scaler = StandardScaler()
  feats = scaler.fit_transform(feats)


  tf.logging.info('Data loaded, %f seconds.', time.time() - start_time)
  #adj,feats都是顺序的
  return num_data, adj, feats, id_map


def preprocess_radeliu(adj,
               features,
               visible_data,
               num_clusters,
               diag_lambda=-1):
  """Do graph partitioning and preprocessing for SGD training."""

  # Do graph partitioning
  part_adj, parts = partition_graph(adj, visible_data,
                                                    num_clusters)

  '''
  if diag_lambda == -1:
    part_adj = normalize_adj(part_adj)
  else:
    part_adj = normalize_adj_diag_enhance(part_adj, diag_lambda)
  '''

  parts = [np.array(pt, dtype=np.int32) for pt in parts]

  print("parts:", parts[0][:100])
  print("features:", features[:100])

  features_batches = []
  support_batches = []

  total_nnz = 0
  for pt in parts:
    features_batches.append(features[pt, :])
    now_part = part_adj[pt, :][:, pt]
    total_nnz += now_part.count_nonzero()
    support_batches.append(now_part)

  return (parts, features_batches, support_batches)


def normalize_adj(adj):
  rowsum = np.array(adj.sum(1)).flatten()
  d_inv = 1.0 / (np.maximum(1.0, rowsum))
  d_mat_inv = sp.diags(d_inv, 0)
  adj = d_mat_inv.dot(adj)
  return adj


def normalize_adj_diag_enhance(adj, diag_lambda):
  """Normalization by  A'=(D+I)^{-1}(A+I), A'=A'+lambda*diag(A')."""
  adj = adj + sp.eye(adj.shape[0])
  rowsum = np.array(adj.sum(1)).flatten()
  d_inv = 1.0 / (rowsum + 1e-20)
  d_mat_inv = sp.diags(d_inv, 0)
  adj = d_mat_inv.dot(adj)
  adj = adj + diag_lambda * sp.diags(adj.diagonal(), 0)
  return adj


def sparse_to_tuple(sparse_mx):
  """Convert sparse matrix to tuple representation."""

  def to_tuple(mx):
    if not sp.isspmatrix_coo(mx):
      mx = mx.tocoo()
    coords = np.vstack((mx.row, mx.col)).transpose()
    values = mx.data
    shape = mx.shape
    return coords, values, shape

  if isinstance(sparse_mx, list):
    for i in range(len(sparse_mx)):
      sparse_mx[i] = to_tuple(sparse_mx[i])
  else:
    sparse_mx = to_tuple(sparse_mx)

  return sparse_mx