#!/usr/bin/python -u
import platform
import json

import os, sys

sys.path.append(os.getcwd())

import time

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from Config import Config
import tflib as lib
from tflib import tianchivect

from match_model_czy import WGAN
from Evaluation import evaluation
from tflib.wgan_saver import wgan_loader, wgan_saver

flags = tf.flags
flags.DEFINE_string("data_path", "./data", "Where the training/test data is stored.")

FLAGS = flags.FLAGS
seed = 2

def INFO_LOG(info):
    print "[%s]%s" % (time.strftime("%Y-%m-%d %X", time.localtime()), info)


def run(session, config, model, flag, reader_s, reader_u=None, verbose=False):
    s_gen = reader_s()

    g_cost = 0.
    d_cost = 0.
    delta = 0.
    orth = 0.
    gen = s_gen
    wdistance = 0.

    def _gen_feed(model, _data):
        # print _data[1], _data[4]
        return {
            model.input1: _data[2],
            model.input_c1: np.array(_data[1]).reshape((1,)) ,
            model.input2: _data[5],
            model.input_c2: np.array(_data[4]).reshape((1,))
        }

    if flag == "Train":
        u_gen = reader_u()

        for iter in xrange(config.ITERS):
            # wdistance = 0.
            if iter < config.ITERS * config.ratio_super[0]:
                gen = s_gen
            else:
                gen = u_gen

            for i in xrange(config.disc_iter):
                _data = gen.next()
                _disc_cost, _orth, _ = session.run(
                    [model.disc_cost, model.orth, model.disc_train_op],
                    feed_dict=_gen_feed(model, _data)
                )
                d_cost += _disc_cost
                orth += _orth
                wdistance -= _disc_cost
                # print "disc", _disc_cost

            for i in xrange(config.gen_iter):
                _data = gen.next()
                _gen_cost, _delta, _orth, _ = session.run(
                    [model.gen_cost, model.delta, model.orth, model.gen_train_op],
                    feed_dict=_gen_feed(model, _data)
                )
                g_cost += _gen_cost
                delta += _delta
                orth += _orth
                wdistance += _gen_cost

    else:
        for iter in xrange(config.ITERS):
            gen = s_gen
            for i in xrange(config.disc_iter):
                _data = gen.next()
                _disc_cost, _orth = session.run(
                    [model.disc_cost, model.orth],
                    feed_dict=_gen_feed(model, _data)
                )
                d_cost += _disc_cost
                orth += _orth
                wdistance -= disc_cost
                # print "disc", _disc_cost

            for i in xrange(config.gen_iter):
                _data = gen.next()
                _gen_cost, _delta, _orth = session.run(
                    [model.gen_cost, model.delta, model.orth],
                    feed_dict=_gen_feed(model, _data)
                )
                g_cost += _gen_cost
                delta += _delta
                orth += _orth
                wdistance += _gen_cost
            # print "gen", _gen_cost
            # print "delta", _delta
    wdistance = wdistance / (float(config.ITERS) * float(config.disc_iter + config.gen_iter))

    return g_cost / (float(config.ITERS) * config.gen_iter), \
           d_cost / float(float(config.disc_iter) * config.ITERS), \
           delta / float(config.ITERS), orth / float(config.ITERS), wdistance


def main(_):
    # Dataset iterator
    config = Config(tianchivect)
    train_s, train_u, test_gen = tianchivect.load(30, 30, seed)
    config.gpuid = 0
    # config = Config()
    def inf_train_gen():
        while True:
            for images in train_gen():
                yield images

    if platform.system() == 'Linux':
        gpuid = config.gpuid
        os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(gpuid)
        device = '/gpu:' + str(gpuid)
    else:
        device = '/cpu:0'

    graph = tf.Graph()
    with graph.as_default():
        trainm = WGAN(config, mode="Train", device=device, reuse=False)
        validm = WGAN(config, mode="Valid", device=device, reuse=True)

    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    allocat_time = 0
    best_auc = 0
    best_acc = 0.
    best_epoch = 0
    epoch_auc_wdist = []
    with tf.Session(graph=graph, config=session_config) as session:
        # writer = tf.summary.FileWriter("./util/", session.graph)
        session.run(tf.global_variables_initializer())
        for epoch in range(config.epoch_num):
            # if epoch > 50:
            #     config.ITERS = 1

            INFO_LOG("Epoch {}".format(epoch))
            g_cost, d_cost, delta, orth, wdistance = run(session, config, trainm, "Train", train_s, train_u,
                                                         verbose=False)
            INFO_LOG("Epoch %d Train g_costs %.5f, d_cost  %.5f, delta  %.5f, orth  %.5f, wdistance %.5f" %
                     (epoch + 1, g_cost, d_cost, delta, orth, wdistance))

            # g_cost, d_cost, delta, orth = run(session, config, validm, "Valid", train_s, verbose=False)
            # INFO_LOG("Epoch %d Test g_costs %.5f, d_cost  %.5f, delta  %.5f, orth  %.5f" %
            #          (epoch + 1, g_cost, d_cost, delta, orth))
            #
            # if epoch % 10 != 1:
            #     INFO_LOG("*** best AUC now is %.3f in %d epoch" % (best_auc, best_epoch))
            # #     continue
            AUC, acc = evaluation(session, config, validm, test_gen, verbose=False)
            INFO_LOG("Epoch %d AUC is %.5f" %
                     (epoch + 1, AUC))
            if best_auc < AUC:
                best_auc = AUC
                best_epoch = epoch
                # wgan_saver(session, config)
            if best_acc < acc:
                best_acc = acc
            epoch_auc_wdist.append((epoch, AUC, wdistance, orth))
            INFO_LOG("*** best AUC now is %.5f in %d epoch" % (best_auc, best_epoch))
            INFO_LOG("*** best acc now is %.5f" % best_acc)
            # if ((epoch +1) % 5 == 0):
            # if ((epoch + 1) % 20 != 0):
            # use another way to train
            # print epoch_auc_wdist
            with open('iter_1000train_sessionauc.json', 'w') as ff:
                ff.write(json.dumps(epoch_auc_wdist))


if __name__ == '__main__':
    tf.app.run()

