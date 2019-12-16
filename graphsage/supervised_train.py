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

from graphsage.supervised_models import SupervisedGraphsage
from graphsage.models import SAGEInfo
from graphsage.minibatch import NodeMinibatchIterator
from graphsage.neigh_samplers import UniformNeighborSampler
from graphsage.utils import load_data

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
# core params..
flags.DEFINE_string('model', 'gat', 'model names. See README for possible values.')  # graphsage_mean,gat,gcn
flags.DEFINE_float('learning_rate', 0.005, 'initial learning rate.')
flags.DEFINE_string("model_size", "small", "Can be big or small; model specific def'ns")
flags.DEFINE_string('train_prefix', "/data/radeliu/ceph/radeliu/campany_relation/20190825XX/campany",
                    'prefix identifying training data. must be specified.')  # /graph_data

# left to default values in main experiments
flags.DEFINE_integer('epochs', 1000, 'number of epochs to train.')
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
flags.DEFINE_boolean('sigmoid', False, 'whether to use sigmoid loss')
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

flags.DEFINE_string('save_name', '/data/radeliu/ceph/radeliu/campany_relation/20190824/model_output/mymodel.ckpt',
                    'Path for saving model')

# os.environ["CUDA_VISIBLE_DEVICES"]=str(FLAGS.gpu)

GPU_MEM_FRACTION = 0.8


def calc_f1(y_true, y_pred):
    if not FLAGS.sigmoid:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)

    # return metrics.f1_score(y_true, y_pred, average="micro"), metrics.f1_score(y_true, y_pred, average="macro")
    # return metrics.f1_score(y_true, y_pred), metrics.recall_score(y_true, y_pred)
    # return metrics.precision_score(y_true, y_pred), metrics.recall_score(y_true, y_pred)
    return metrics.roc_auc_score(y_true, y_pred), max(tpr - fpr)


# Define model evaluation function
def evaluate(sess, merged, model, minibatch_iter, adj_info_ph, feature_info_ph, features, size=None):
    t_test = time.time()
    feed_dict_val, labels = minibatch_iter.node_val_feed_dict(size)
    feed_dict_val.update({adj_info_ph: minibatch_iter.test_adj})
    feed_dict_val.update({feature_info_ph: features})

    node_outs_val = sess.run([model.preds, model.loss, model.auc_op, model.auc_value, merged],
                             feed_dict=feed_dict_val)
    mic, mac = calc_f1(labels, node_outs_val[0])
    return node_outs_val[1], node_outs_val[-2], mic, mac, (time.time() - t_test), node_outs_val[-1]


def log_dir():
    log_dir = FLAGS.base_log_dir + "/sup-" + FLAGS.train_prefix.split("/")[-2]
    # log_dir += "/{model:s}_graphsage_concat_{model_size:s}_{lr:0.4f}/".format(
    # log_dir += "/{model:s}_only_gat_bilstm_{model_size:s}_{lr:0.4f}/".format(
    log_dir += "/{model:s}_attentionweight_{model_size:s}_{lr:0.4f}/".format(
        model=FLAGS.model,
        model_size=FLAGS.model_size,
        lr=FLAGS.learning_rate)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def incremental_evaluate(sess, model, minibatch_iter, adj_info_ph, feature_info_ph, features, size, test=False):
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
    return np.mean(val_losses), f1_scores[0], f1_scores[1], (time.time() - t_test)


def construct_placeholders(num_classes):
    # Define placeholders
    placeholders = {
        'labels': tf.placeholder(tf.float32, shape=(None, num_classes), name='labels'),
        'batch': tf.placeholder(tf.int32, shape=(None), name='batch1'),
        'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
        'batch_size': tf.placeholder(tf.int32, name='batch_size'),
    }
    return placeholders


def train(train_data, test_data=None):
    # train_data[3]是walks
    G = train_data[0]
    features = train_data[1]
    id_map = train_data[2]
    class_map = train_data[3]

    # print('class_map.values():',class_map.values())

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

    if isinstance(list(class_map.values())[0], list):
        num_classes = len(list(class_map.values())[0])
    else:
        num_classes = len(set(class_map.values()))

    print("num_classes:", num_classes)

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
                                      max_degree=FLAGS.max_degree)

    adj_info_ph = tf.placeholder(tf.int32, shape=(None, None), name="adj_info")  # minibatch.adj.shape
    adj_info = adj_info_ph

    feature_info_ph = tf.placeholder(tf.float32, shape=(None, features.shape[1]), name="feature_info")

    # adj_info_ph = tf.placeholder(tf.int32, shape=minibatch.adj.shape)
    # adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info")

    val_class_test = [class_map[id] for id in minibatch.val_nodes]
    print("val_class_test:", val_class_test)
    val_class = [np.argmax(class_map[id], axis=0) for id in minibatch.val_nodes]
    val_id_map = [id_map[id] for id in minibatch.val_nodes]
    val_feature = features[val_id_map]

    train_class = [np.argmax(class_map[id], axis=0) for id in minibatch.train_nodes]
    train_id_map = [id_map[id] for id in minibatch.train_nodes]
    train_feature = features[train_id_map]

    np.savetxt(log_dir() + "val_data.txt",
               np.around(np.hstack((np.array(val_class).reshape(-1, 1), np.array(val_feature))), decimals=2))
    np.savetxt(log_dir() + "train_data.txt",
               np.around(np.hstack((np.array(train_class).reshape(-1, 1), np.array(train_feature))), decimals=2))

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
                                    features.shape[1],
                                    concat=False,
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
                                    layer_infos,
                                    features.shape[1],
                                    aggregator_type="gcn",
                                    model_size=FLAGS.model_size,
                                    concat=False,
                                    sigmoid_loss=FLAGS.sigmoid,
                                    identity_dim=FLAGS.identity_dim,
                                    logging=True)

    elif FLAGS.model == 'graphsage_seq':
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                       SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SupervisedGraphsage(num_classes, placeholders,
                                    features,
                                    adj_info,
                                    minibatch.deg,
                                    layer_infos=layer_infos,
                                    aggregator_type="seq",
                                    model_size=FLAGS.model_size,
                                    sigmoid_loss=FLAGS.sigmoid,
                                    identity_dim=FLAGS.identity_dim,
                                    logging=True)

    elif FLAGS.model == 'graphsage_maxpool':
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                       SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SupervisedGraphsage(num_classes, placeholders,
                                    features,
                                    adj_info,
                                    minibatch.deg,
                                    layer_infos=layer_infos,
                                    aggregator_type="maxpool",
                                    model_size=FLAGS.model_size,
                                    sigmoid_loss=FLAGS.sigmoid,
                                    identity_dim=FLAGS.identity_dim,
                                    logging=True)

    elif FLAGS.model == 'graphsage_meanpool':
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                       SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SupervisedGraphsage(num_classes, placeholders,
                                    features,
                                    adj_info,
                                    minibatch.deg,
                                    layer_infos=layer_infos,
                                    aggregator_type="meanpool",
                                    model_size=FLAGS.model_size,
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

    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = GPU_MEM_FRACTION
    config.allow_soft_placement = True

    # Initialize session
    sess = tf.Session(config=config)
    merged = tf.summary.merge_all()
    # summary_writer = tf.summary.FileWriter(log_dir(), sess.graph)
    summary_writer = tf.summary.FileWriter(log_dir())

    tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES)

    # Init variables
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    saver = tf.train.Saver()

    # Train model

    total_steps = 0
    avg_time = 0.0
    epoch_val_costs = []

    print("minibatch.adj.shape:", minibatch.adj.shape)
    print("minibatch.test_adj.shape:", minibatch.test_adj.shape)

    # train_adj_info = tf.assign(adj_info, minibatch.adj)
    # val_adj_info = tf.assign(adj_info, minibatch.test_adj)

    metric_output = []

    for epoch in range(FLAGS.epochs):

        print("train,val,test amount:",
              (len(minibatch.train_nodes), len(minibatch.val_nodes), len(minibatch.test_nodes)))
        print("train nodes:", minibatch.train_nodes)

        minibatch.shuffle()

        iter = 0
        print('Epoch: %04d' % (epoch + 1))
        epoch_val_costs.append(0)
        while not minibatch.end():
            print("total_steps,iter:", (total_steps, iter))
            # Construct feed dictionary
            feed_dict, labels = minibatch.next_minibatch_feed_dict()
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            feed_dict.update({adj_info_ph: minibatch.adj})
            feed_dict.update({feature_info_ph: features})

            t = time.time()
            # Training step
            outs = sess.run([merged, model.opt_op, model.loss, model.auc_op, model.auc_value, model.preds],
                            feed_dict=feed_dict)
            train_cost = outs[2]

            if iter % FLAGS.validate_iter == 0:
                # Validation
                # sess.run(val_adj_info.op)
                if FLAGS.validate_batch_size == -1:
                    val_cost, val_f1_mic, val_f1_mac, duration = incremental_evaluate(sess, model, minibatch,
                                                                                      FLAGS.batch_size)
                else:
                    val_cost, auc, val_f1_mic, val_f1_mac, duration, merged_value = evaluate(sess, merged, model,
                                                                                             minibatch, adj_info_ph,
                                                                                             feature_info_ph, features,
                                                                                             FLAGS.validate_batch_size)
                # sess.run(train_adj_info.op)
                epoch_val_costs[-1] += val_cost

            if total_steps % FLAGS.print_every == 0:
                summary_writer.add_summary(merged_value, total_steps)
                summary_writer.flush()

            # Print results
            avg_time = (avg_time * total_steps + time.time() - t) / (total_steps + 1)

            if total_steps % FLAGS.print_every == 0:
                # print("labels_before:", labels)
                # print("predict_before:", outs[-1])
                # print("labels:", np.argmax(labels, axis=1))
                # print("predict:", np.argmax(outs[-1], axis=1))
                train_f1_mic, train_f1_mac = calc_f1(labels, outs[-1])
                print("Iter:", '%04d' % iter,
                      "train_loss=", "{:.5f}".format(train_cost),
                      "train_f1_mic=", "{:.5f}".format(train_f1_mic),
                      "train_f1_mac=", "{:.5f}".format(train_f1_mac),
                      "val_loss=", "{:.5f}".format(val_cost),
                      "val_f1_mic=", "{:.5f}".format(val_f1_mic),
                      "val_f1_mac=", "{:.5f}".format(val_f1_mac),
                      "time=", "{:.5f}".format(avg_time))
                metric_output.append([total_steps + 1, val_cost, val_f1_mic, val_f1_mac, auc, outs[-3]])

            stream_vars_valid = [v for v in tf.local_variables()]
            sess.run(tf.variables_initializer(stream_vars_valid))

            iter += 1
            total_steps += 1

            if total_steps > FLAGS.max_total_steps:
                break

        if total_steps > FLAGS.max_total_steps:
            break

    print("Optimization Finished!")

    np.savetxt(log_dir() + "metric_output.txt", np.around(np.array(metric_output).reshape(-1, 6), decimals=3))

    # sess.run(val_adj_info.op)
    val_cost, val_f1_mic, val_f1_mac, duration = incremental_evaluate(sess, model, minibatch, adj_info_ph,
                                                                      feature_info_ph, features, FLAGS.batch_size)
    print("Full validation stats:",
          "loss=", "{:.5f}".format(val_cost),
          "f1_micro=", "{:.5f}".format(val_f1_mic),
          "f1_macro=", "{:.5f}".format(val_f1_mac),
          "time=", "{:.5f}".format(duration))
    with open(log_dir() + "val_stats.txt", "w") as fp:
        fp.write("loss={:.5f} f1_micro={:.5f} f1_macro={:.5f} time={:.5f}".
                 format(val_cost, val_f1_mic, val_f1_mac, duration))

    print("Writing test set stats to file (don't peak!)")
    val_cost, val_f1_mic, val_f1_mac, duration = incremental_evaluate(sess, model, minibatch, adj_info_ph,
                                                                      feature_info_ph, features, FLAGS.batch_size,
                                                                      test=True)
    with open(log_dir() + "test_stats.txt", "w") as fp:
        fp.write("loss={:.5f} f1_micro={:.5f} f1_macro={:.5f}".
                 format(val_cost, val_f1_mic, val_f1_mac))

    saver.save(sess, FLAGS.save_name)


def main(argv=None):
    print("Loading training data..")
    train_data = load_data(FLAGS.train_prefix)
    print("Done loading training data..")
    train(train_data)


if __name__ == '__main__':
    tf.app.run()
