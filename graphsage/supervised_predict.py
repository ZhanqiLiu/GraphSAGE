#!/data/anaconda3/bin/python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import os
import time
import tensorflow as tf
import numpy as np
import sklearn
from sklearn import metrics
import networkx as nx
import os
from networkx.readwrite import json_graph
import json

os.environ["METIS_DLL"] = "/data/radeliu/ceph/radeliu/data_gcn/libmetis.so"

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean('log_device_placement', False,
                     """Whether to log device placement.""")
# core params..
flags.DEFINE_string('model', 'gat', 'model names. See README for possible values.')  # gat,graphsage_mean
flags.DEFINE_float('learning_rate', 0.005, 'initial learning rate.')
flags.DEFINE_string("model_size", "small", "Can be big or small; model specific def'ns")
flags.DEFINE_string('train_prefix', "/data/radeliu/ceph/radeliu/campany_relation/predict/",
                    'prefix identifying training data. must be specified.')

# left to default values in main experiments
flags.DEFINE_integer('epochs', 1, 'number of epochs to train.')
flags.DEFINE_float('dropout', 0.1, 'dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0.1, 'weight for l2 loss on embedding matrix.')
flags.DEFINE_integer('max_degree', 20, 'maximum node degree.')
flags.DEFINE_integer('samples_1', 20, 'number of samples in layer 1')
flags.DEFINE_integer('samples_2', 20, 'number of samples in layer 2')
flags.DEFINE_integer('samples_3', 0, 'number of users samples in layer 3. (Only for mean model)')
flags.DEFINE_integer('dim_1', 31, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_integer('dim_2', 31, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_boolean('random_context', True, 'Whether to use random context or direct edges')
flags.DEFINE_integer('batch_size', 2048, 'minibatch size.')
flags.DEFINE_boolean('sigmoid', False, 'whether to use sigmoid loss')  # sigmoid(x)=1/1+e-x
flags.DEFINE_integer('identity_dim', 0,
                     'Set to positive value to use identity embedding features of that dimension. Default 0.')

# logging, saving, validation settings etc.
flags.DEFINE_string('base_log_dir', '/data/radeliu/ceph/radeliu/sagelog',
                    'base directory for logging and saving embeddings')
flags.DEFINE_integer('validate_iter', 1, "how often to run a validation minibatch.")
flags.DEFINE_integer('validate_batch_size', 1746, "how many nodes per validation sample.")
flags.DEFINE_integer('gpu', 1, "which gpu to use.")
flags.DEFINE_integer('print_every', 1, "How often to print training info.")
flags.DEFINE_integer('max_total_steps', 10 ** 10, "Maximum total number of iterations")

flags.DEFINE_string('save_name', '/data/radeliu/ceph/radeliu/campany_relation/model_output/mymodel.ckpt',
                    'Path for saving model')
flags.DEFINE_integer('num_clusters_test', 10, 'Number of clusters for test.')
flags.DEFINE_float('diag_lambda', 1,
                   'A positive number for diagonal enhancement, -1 indicates normalization without diagonal enhancement')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)
GPU_MEM_FRACTION = 0.8


# Define model evaluation function


def calc_f1(y_true, y_pred):
    if not FLAGS.sigmoid:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
    # fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)

    auc_tensor = tf.convert_to_tensor(metrics.f1_score(y_true, y_pred, average="micro"))
    tf.summary.scalar('f1_score_direct', auc_tensor)

    return metrics.f1_score(y_true, y_pred, average="micro"), metrics.f1_score(y_true, y_pred,
                                                                               average="macro")  # micro,整体算TP，FN , FP，然后算F1；macro按每个类别分别算，然后（F1+F2+F3+F4）/4
    # return metrics.precision_score(y_true, y_pred), metrics.recall_score(y_true, y_pred)
    # return metrics.roc_auc_score(y_true, y_pred, max(tpr - fpr)


# Define model evaluation function
def evaluate(sess, merged, model, minibatch_iter, size=None):
    t_test = time.time()
    feed_dict_val, labels = minibatch_iter.node_val_feed_dict(size)
    node_outs_val = sess.run([model.preds, model.loss, merged],
                             feed_dict=feed_dict_val)
    mic, mac = calc_f1(labels, node_outs_val[0])
    return node_outs_val[1], mic, mac, (time.time() - t_test), node_outs_val[-1]


def log_dir():
    log_dir = FLAGS.base_log_dir + "/sup-" + FLAGS.train_prefix.split("/")[-2]
    log_dir += "/{model:s}_graphsage_concat_{model_size:s}_{lr:0.4f}/".format(
        model=FLAGS.model,
        model_size=FLAGS.model_size,
        lr=FLAGS.learning_rate)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def incremental_evaluate(sess, model, minibatch_iter,adj_info_ph,feature_info_ph,features,  size, test=False):
    t_test = time.time()
    finished = False
    val_losses = []
    val_preds = []
    labels = []
    iter_num = 0
    finished = False
    while not finished:
        feed_dict_val, batch_labels, finished, _ = minibatch_iter.incremental_node_val_feed_dict(size, iter_num,
                                                                                                 test=test)
        feed_dict_val.update({adj_info_ph: minibatch_iter.test_adj})
        feed_dict_val.update({feature_info_ph: features})

        node_outs_val = sess.run([model.preds, model.loss],
                                 feed_dict=feed_dict_val)
        val_preds.append(node_outs_val[0])
        labels.append(batch_labels)
        val_losses.append(node_outs_val[1])
        iter_num += 1
    val_preds = np.vstack(val_preds)
    labels = np.vstack(labels)
    f1_scores = calc_f1(labels, val_preds)
    #return np.mean(val_losses), f1_scores[0], f1_scores[1], (time.time() - t_test)
    return np.argmax(val_preds, axis=1)


def construct_placeholders(num_classes):
    # Define placeholders
    placeholders = {
        'labels': tf.placeholder(tf.float32, shape=(None, num_classes), name='labels'),
        'batch': tf.placeholder(tf.int32, shape=(None), name='batch1'),
        'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
        'batch_size': tf.placeholder(tf.int32, name='batch_size'),
    }
    return placeholders

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


def main_app(argv=None):
    try:
        from pip import main
    except Exception as e:
        from pip._internal import main
    except Exception as e:
        from pip.__main__ import _main as main

    main(["install", "metis"])

    from graphsage.supervised_models import SupervisedGraphsage
    from graphsage.models import SAGEInfo
    from graphsage.minibatch import NodeMinibatchIterator
    from graphsage.neigh_samplers import UniformNeighborSampler
    from graphsage.utils import load_predict_data, preprocess_radeliu

    print("Loading predicting data..")
    predict_data = load_predict_data(FLAGS.train_prefix)
    print("Done loading predicting data..")

    num_data = predict_data[0]
    adj = predict_data[1]
    feats = predict_data[2]
    id_map_all = predict_data[3]
    id_map_all_reversed = {v: k for k, v in id_map_all.items()}

    print("data shape:", len(id_map_all))

    (parts, test_features_batches, test_support_batches) = preprocess_radeliu(adj, feats,
                                                                              np.arange(num_data),
                                                                              FLAGS.num_clusters_test,
                                                                              FLAGS.diag_lambda)

    print("parts length:", [len(i) for i in parts])
    print("test_features_batches length:", len(test_features_batches))
    print("test_support_batches length:", len(test_support_batches))

    #test_support_batches_output=[adj.toarray for adj in test_support_batches]

    #np.savez('/data/radeliu/ceph/radeliu/campany_relation/predict/parts.npz', parts)
    #np.savez('/data/radeliu/ceph/radeliu/campany_relation/predict/test_features_batches.npz', test_features_batches)
    #np.savez('/data/radeliu/ceph/radeliu/campany_relation/predict/test_support_batches.npz', test_support_batches_output)

    load_model_flag = 1

    for part_id in range(len(parts)):



        G = nx.from_scipy_sparse_matrix(test_support_batches[part_id])

        test_index = list(range(test_features_batches[part_id].shape[0]))
        y_test = np.random.randint(0, 2, len(test_index)).reshape((len(test_index), 1))

        classMap = {}
        classmap_test = dict(zip(range(len(test_index)), y_test))
        classMap.update(classmap_test)

        really_key_part = [id_map_all_reversed[j] for j in parts[part_id]]

        idMap = {}
        idmap_test = dict(zip(test_index, test_index))
        idMap.update(idmap_test)

        print("length:", len(y_test))

        for i in range(len(y_test)):
            G.node[i]['test'] = True
            G.node[i]['val'] = False


        features = test_features_batches[part_id]
        id_map = idMap
        class_map = classMap


        def onehotlabel(x):
            if x == [0]:
                # print("white:",(x,[1,0]))
                return [1, 0]
            else:
                # print("black:",(x,[0,1]))
                return [0, 1]

        class_map = {k: onehotlabel(v) for k, v in class_map.items()}

        print("data shape:", features.shape, len(id_map), len(class_map))

        print("class_map[0]:", class_map[0], class_map[1])

        print("data shape:", features.shape, len(id_map), len(class_map))
        if isinstance(list(class_map.values())[0], list):
            num_classes = len(list(class_map.values())[0])
        else:
            num_classes = len(set(class_map.values()))

        if not features is None:
            # pad with dummy zero vector
            features = np.vstack([features, np.zeros((features.shape[1],))])

        placeholders = construct_placeholders(num_classes)
        minibatch = NodeMinibatchIterator(G,
                                          id_map,
                                          placeholders,
                                          class_map,
                                          num_classes,
                                          batch_size=FLAGS.batch_size,
                                          max_degree=FLAGS.max_degree,
                                          train_flag=False)

        #adj_info_ph = tf.placeholder(tf.int32, shape=minibatch.adj.shape)
        #adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info")

        adj_info_ph = tf.placeholder(tf.int32, shape=(None, None), name="adj_info")#minibatch.adj.shape
        adj_info = adj_info_ph

        feature_info_ph = tf.placeholder(tf.float32, shape=(None, features.shape[1]), name="feature_info")


        if FLAGS.model == 'graphsage_mean':
            # Create model
            sampler = UniformNeighborSampler(adj_info)  # adj_info进行初始化，后面调用的时候需要给id,number
            if FLAGS.samples_3 != 0:
                layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                               SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2),
                               SAGEInfo("node", sampler, FLAGS.samples_3, FLAGS.dim_2)]
            elif FLAGS.samples_2 != 0:
                layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                               SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]
            else:
                layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1)]

            model = SupervisedGraphsage(num_classes, placeholders,
                                        feature_info_ph,
                                        adj_info,
                                        minibatch.deg,
                                        layer_infos,
                                        concat=True,
                                        model_size=FLAGS.model_size,
                                        sigmoid_loss=FLAGS.sigmoid,
                                        identity_dim=FLAGS.identity_dim,
                                        logging=True)
        elif FLAGS.model == 'gcn':
            # Create model
            sampler = UniformNeighborSampler(adj_info)
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, 2 * FLAGS.dim_1),
                           SAGEInfo("node", sampler, FLAGS.samples_2, 2 * FLAGS.dim_2)]

            model = SupervisedGraphsage(num_classes, placeholders,
                                        feature_info_ph,
                                        adj_info,
                                        minibatch.deg,
                                        layer_infos=layer_infos,
                                        aggregator_type="gcn",
                                        model_size=FLAGS.model_size,
                                        concat=False,
                                        sigmoid_loss=FLAGS.sigmoid,
                                        identity_dim=FLAGS.identity_dim,
                                        logging=True)

        elif FLAGS.model == 'gat':
            # Create model
            sampler = UniformNeighborSampler(adj_info)  # adj_info进行初始化，后面调用的时候需要给id,number
            if FLAGS.samples_3 != 0:
                layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                               SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2),
                               SAGEInfo("node", sampler, FLAGS.samples_3, FLAGS.dim_2)]
            elif FLAGS.samples_2 != 0:
                layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                               SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]
            else:
                layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1)]

            model = SupervisedGraphsage(num_classes, placeholders,
                                        feature_info_ph,
                                        adj_info,
                                        minibatch.deg,
                                        layer_infos,
                                        features.shape[1],
                                        concat=False,
                                        aggregator_type=FLAGS.model,
                                        model_size=FLAGS.model_size,
                                        sigmoid_loss=FLAGS.sigmoid,
                                        identity_dim=FLAGS.identity_dim,
                                        logging=True)

        else:
            raise Exception('Error: model name unrecognized.')

        if load_model_flag == 1:
            config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)

            # TensorFlow程序会默认占用显卡中的所有显存，如果想让程序需要多少显存就用多少应该怎么设置呢
            config.gpu_options.allow_growth = True
            # config.gpu_options.per_process_gpu_memory_fraction = GPU_MEM_FRACTION
            config.allow_soft_placement = True

            # Initialize session
            sess = tf.Session(config=config)

            #sess.run(tf.global_variables_initializer(), feed_dict={adj_info_ph: minibatch.adj})
            #sess.run(tf.local_variables_initializer())

            tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES)

            saver = tf.train.Saver()
            #saver.restore(sess, FLAGS.save_name)#此处应该可以删除掉
            load_model_flag = 0

        sess.run(tf.global_variables_initializer(), feed_dict={adj_info_ph: minibatch.adj})
        sess.run(tf.local_variables_initializer())
        saver.restore(sess, FLAGS.save_name)
        print("Writing test set stats to file (don't peak!)")
        val_preds = incremental_evaluate(sess, model, minibatch,adj_info_ph,feature_info_ph,features,FLAGS.batch_size, test=True)

        predict_result = dict(zip(really_key_part, val_preds))

        with open('/data/radeliu/ceph/radeliu/campany_relation/predict' + "/" + "predict_output_{:03d}.json".format(part_id+1), "w") as f:
            json.dump(predict_result, f, cls=MyEncoder)

    try:
        test_support_batches_output = [adj.toarray() for adj in test_support_batches]
        np.savez('/data/radeliu/ceph/radeliu/campany_relation/predict/parts.npz', parts)
        np.savez('/data/radeliu/ceph/radeliu/campany_relation/predict/test_features_batches.npz', test_features_batches)
        np.savez('/data/radeliu/ceph/radeliu/campany_relation/predict/test_support_batches.npz',test_support_batches_output)
    except:
        pass

if __name__ == '__main__':
    tf.app.run(main_app)