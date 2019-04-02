import tensorflow as tf
from collections import OrderedDict
import json
import math
import numpy as np
import random
import os

from tflib.tsne import tsne, draw_image
"""
i use AUC to evaluate this problem just because the former method is use this kind of metrice

as i think this problem should use something likes MAP to evalutae 
"""


def generate_tsne(session, config, model, test_gen, catelist, verbose=False):
    """
    the test list should be the same as the other method, use a triple tuple
    if x_{q,i}, x_{q,j}

    """
    print "generate_sne"
    vectorpath = '../../../../../mnt/dev_sdb1/cuizeyu/image_vector/'
    imgpath = '../../../../../mnt/dev_sdb1/cuizeyu/tianchi_fm_img/'
    filepath = './data/'
    vectorpath = '/home/cuizeyu/image_vector/'
    # stylepath = './style_vector/'
    # "first generate all the style vector we need"
    style_dict_full = {}
    for cate in catelist:
        if not os.path.exists(stylepath + 'style_dict_{}.json' % str(cate)):
            style_dict = OrderedDict()
            ttt = 0
            for item_id, c_input, im_input in test_gen():
                ttt += 1
                if ttt > 500:
                    break
                style_vec = session.run(
                    model.goutput,
                    feed_dict={model.ginput: im_input, model.ginput_c:np.array(c_input).reshape((1,))}
                )
                # print style_vec.shape
                for idx in xrange(config.BATCH_SIZE):
                    style_dict[item_id[idx]] = style_vec[idx].tolist()
                # for item, sty_v in zip(*(item_id, style_vec)):
                #     style_dict[item] = sty_v
            with open(stylepath + 'style_dict_{}.json' % str(cate), 'w') as f:
                f.write(json.dumps(style_dict.items()))
        else:
            with open(stylepath + 'style_dict_{}.json' % str(cate), 'r') as f:
                style_dict = dict(json.load(f))
        style_dict_full.update(d1)


    styy_random = random.sample(style_dict_full.items(), 2500)

    labels, X = zip(*styy_random)
    print labels[0], X[0]
    cal_and_draw(labels, X, imgpath, 'style_img.png')

    # draw the origin map as a compairsion
    vect_batch = []
    for name in labels:
        with open(vectorpath + name + '.json') as f:
            vect_batch.append(json.load(f))

    cal_and_draw(labels, np.asarray(vect_batch), imgpath,'vgg_img.png')



def sigmoid(x):
    return 1./(1. + np.exp(-x))

def cal_and_draw(labels, X, imgpath, img_name):
    X = np.asarray(X)
    X = X / max([X.max(), abs(X.min())])

    Y = tsne(X, 2, 50, 20.0)

    draw_image(Y, labels, imgpath, img_name)