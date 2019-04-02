import tensorflow as tf
from collections import OrderedDict
import json
import math
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import random
from PIL import Image
import imghdr
import urllib2
import os


"""
i use AUC to evaluate this problem just because the former method is use this kind of metrice

as i think this problem should use something likes MAP to evalutae 
"""

# filepath = './data/'
# vectorpath = '/home/cuizeyu/image_vector/'


filepath = './data/clothing/Women/'
vectorpath = '/home/cuizeyu/amazon_image_feature/clothing/Women/'


def evaluation(session, config, model, test_gen, verbose=False):
    """
    the test list should be the same as the other method, use a triple tuple
    if x_{q,i}, x_{q,j}

    """
    "first generate all the style vector we need"
    style_dict = OrderedDict()
    ttt = 0
    for item_id, c_input, im_input in test_gen():
        ttt += 1
        # if ttt % 100 == 0:
        #     print ttt
        style_vec = session.run(
            model.goutput,
            feed_dict={model.ginput: im_input, model.ginput_c: np.array(c_input).reshape((1,))}
        )
        # print style_vec.shape
        for idx in xrange(config.BATCH_SIZE):
            style_dict[item_id[idx]] = style_vec[idx]
            # for item, sty_v in zip(*(item_id, style_vec)):
            #     style_dict[item] = sty_v
    with open(filepath + 'item_category.json', 'r') as f:
        item_category = dict(json.load(f))
    with open(filepath + 'valid_taobao.json', 'r') as f:
        test_data = json.load(f)
    q, p, n = zip(*test_data)
    matchset = set(zip(*(q, p)))
    with open(filepath + "valid_bycate.json", 'r') as f:
        test_dict = dict(json.load(f))

    with open(filepath + "cate_2match.json", 'r') as f:
        cate_2match = json.load(f)
        cate_2match = zip(*(zip(*cate_2match)))
        cate_2match = set(cate_2match)

    def _cate_match(x, y):
        c_x = item_category[x]
        c_y = item_category[y]
        if (c_x, c_y) in cate_2match or (c_y, c_x) in cate_2match:
            return True
        else:
            return False

    def _choice_neg(q, c):
    	n = random.choice(test_dict[c])
    	while (q, n) in matchset or (n, q) in matchset:
    		n = random.choice(test_dict[c])
    	return n
    # q = u'2685301'
    # p = u'1196258'
    # cate_q = item_category[q]
    # cate_p = item_category[p]
    ss = 0.
    tt = 0.
    # setitem = set([u'368', u'111'])
    ttt = 0
    y_test = []
    y_pred = []
    print len(test_data)
    xxx = []
    for query_item, positive_item, negative_item in test_data:
        if not style_dict.has_key(query_item) or not style_dict.has_key(positive_item):
            print "!!!!!!!!!!!!!!!!!"
            print item_category[query_item]
            print item_category[positive_item]
            continue
        x_qp = distance(style_dict[query_item], style_dict[positive_item])
        x_qn = distance(style_dict[query_item], style_dict[negative_item])
        xxx.append(x_qp)
        xxx.append(x_qn)

    x_max = max(xxx)
    x_min = min(xxx)
    print x_max, x_min
    for query_item, positive_item, negative_item in test_data:            
        if not style_dict.has_key(query_item) or not style_dict.has_key(positive_item):
            print "!!!!!!!!!!!!!!!!!"
            continue
        ss += 1.
        x_qp = distance(style_dict[query_item], style_dict[positive_item])
        x_qn = distance(style_dict[query_item], style_dict[negative_item])
        if not _cate_match(query_item, negative_item):
            x_qn = x_max  # (just may be the most largest)
        if x_qp < x_qn:
            tt += 1.
        y_test.append(1)
        y_test.append(0)
        y_pred.append(1. / x_qp)
        y_pred.append(1. / x_qn)
    p_max = max(y_pred)
    p_min = min(y_pred)

    y_pred = map(lambda x: (x - p_min) / float(p_max - p_min), y_pred)

    c = 0.5
    y_pred = map(lambda x: 1. / (1 + math.e ** (-x + c)), y_pred)
    # the sklearn AUC
    acc, max_s = cal_acc(y_test, y_pred)

    roc_auc = roc_auc_score(y_test, y_pred)
    # acc = cal_acc(y_test, y_pred)
    print "the sklearn AUC is here!!!", roc_auc
    print "**and the acc is ", acc
    # for a, b in zip(*(y_test, y_pred)):
    #   print a, b
    AUC = tt / ss
    # print ss, len(test_data)
    return AUC, acc


def distance(x, y):
    return math.sqrt(sum((x - y) ** 2)) / float(len(x))


def cal_acc(y_test, y_pred):
    # y_pred is around from 0-1
    max_acc = 0.0
    max_s = 0.0
    for s in range(30, 60, 1):
        ss = s * 0.01
        yy_pred = map(lambda x: x > ss, y_pred)
        acc = accuracy_score(y_test, yy_pred)

        if acc > max_acc:
            max_acc = acc
            max_s = ss
    yy_pred = map(lambda x: x > max_s, y_pred)
    print 'test acc', accuracy_score2(y_test, yy_pred)
    # yy_pred = map(lambda x: x > max_s, y_pred)
    # print 'test auc', roc_auc_score(y_test, yy_pred)
    print 'max s', max_s
    return max_acc, max_s


def accuracy_score2(y_test, yy_pred):
    ss = 0.
    tt = 0.
    for t, p in zip(*(y_test, yy_pred)):
        ss += 1.
        if t > 0 and p > 0:
            tt += 1.
        if t == 0 and p == 0:
            tt += 1.
    return tt / ss


    # def sigmoid_dist(x, y):
    #   d = distance(x,y)
    #   return 1./(1 + math.e ** (-d))