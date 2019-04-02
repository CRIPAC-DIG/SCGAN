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
from tflib import tianchivect_men as tianchivect

from match_model_czy import WGAN
from Evaluation_men import evaluation
from tflib.wgan_saver import wgan_loader, wgan_saver

flags = tf.flags
flags.DEFINE_string("data_path", "./data", "Where the training/test data is stored.")

FLAGS = flags.FLAGS
seed = 2

def INFO_LOG(info):
    print "[%s]%s" % (time.strftime("%Y-%m-%d %X", time.localtime()), info)


filepath = './data/clothing/Men/'
vectorpath = '/home/cuizeyu/amazon_image_feature/clothing/Men/'
with open(filepath + "item_category.json", 'r') as f:
    item_category = json.load(f)
with open(filepath + 'cate2id.json', 'r') as f:
    cate2id = json.load(f)

def find_item_vec(itemlist):
    length = len(itemlist)
    with open(vectorpath + itemlist[0] + '.json') as f:
        vec = [json.load(f)] * length
    return np.asarray(vec).reshape((length, 4096))

def fin_item_cate(item):

    return cate2id[item_category[item]]

def main(_):
    # Dataset iterator
    config = Config(tianchivect)
    train_s, train_u, test_gen = tianchivect.load(30, 30, seed)
    config.gpuid = 2
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
        wgan_loader(session, config, './save/')

        for epoch in range(config.epoch_num):
            # if epoch > 50:
            #     config.ITERS = 1

            # INFO_LOG("Epoch {}".format(epoch))
            # g_cost, d_cost, delta, orth, wdistance = run(session, config, trainm, "Train", train_s, train_u,
            #                                              verbose=False)
            # INFO_LOG("Epoch %d Train g_costs %.5f, d_cost  %.5f, delta  %.5f, orth  %.5f, wdistance %.5f" %
            #          (epoch + 1, g_cost, d_cost, delta, orth, wdistance))

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
            INFO_LOG("*** best AUC now is %.5f in %d epoch" % (best_auc, best_epoch))
            INFO_LOG("*** best acc now is %.5f" % best_acc)
            # if ((epoch +1) % 5 == 0):
            # if ((epoch + 1) % 20 != 0):
            # use another way to train
            # print epoch_auc_wdist


            # =============================================================== #
            # ==========generate all the style vectors we need ============== #
            # =============================================================== #
            ttt = 0
            item_style_vec = {}
            with open("sixstyle_clothing.json", 'r') as f:
                itemlist_all = json.load(f)
            for itemlist in itemlist_all:
                print len(itemlist)
                for item in itemlist:
                    items = [item] * config.BATCH_SIZE
                    im_input = find_item_vec(items)
                    c_input = fin_item_cate(item)
                    ttt += 1
                    if ttt % 100 == 0:
                        print ttt
                    style_vec = session.run(
                        validm.goutput,
                        feed_dict={validm.ginput: im_input, validm.ginput_c: np.array(c_input).reshape((1,))}
                    )
                    item_style_vec[item] = map(float, list(style_vec[0]))
            with open("item_style_vec.json", 'w') as f:
                f.write(json.dumps(item_style_vec))


if __name__ == '__main__':
    tf.app.run()

