import numpy as np
import random
import os
import itertools
import collections
import json
import urllib

# filepath = '../../../../../mnt/dev_sdb1/cuizeyu/tianchi_fm_img/'
# filepath = '/Users/czy_yente/Downloads/ali_data/item/'
# filepath = './data/'
# vectorpath = '/home/cuizeyu/image_vector/'
filepath = './data/clothing/Girls/'
vectorpath = '/home/cuizeyu/amazon_image_feature/clothing/Girls/'


def trans_into_dict(a):
    key, value = zip(*a)
    key = zip(*zip(*key))
    return dict(zip(*(key, value)))


with open(filepath + "item_category.json", 'r') as f:
    item_category = json.load(f)
catelist = list(set(item_category.values()))
if not os.path.exists(filepath + 'cate2id.json'):
    cate2id = dict(zip(*(catelist, range(len(catelist)))))
    with open(filepath + 'cate2id.json', 'w') as f:
        f.write(json.dumps(cate2id))
        print "generate cate2id"
else:
    with open(filepath + 'cate2id.json', 'r') as f:
        cate2id = json.load(f)
# break
with open(filepath + "train_bycate.json", 'r') as f:
    cate_2match = trans_into_dict(json.load(f)).keys()
# with open(filepath + "cate_2match.json", 'r') as f:
    # cate_2match = json.load(f)


def tianchi_generator(data, batchsize, filepath, vectorpath, seed, flag):
    length = len(data)
    num_batches = length / batchsize
    # with open(filepath + "item_category.json", 'r') as f:
    #     item_category = json.load(f)
    # catelist = list(set(item_category.values()))
    # cate2id = dict(zip(*(catelist, range(len(catelist)))))
    if vectorpath == '/home/cuizeyu/image_vector/':
        supervise_path = filepath + "train_taobao.json"
    else:
        supervise_path = filepath + "seed_" + str(seed)[-1] + "train_taobao.json"
    def supervise_get_epoch():
        with open(supervise_path, 'r') as f:
            train_taobao = json.load(f)

        while True:
            (q, p, n) = random.choice(train_taobao)
            # q = u'2685301'
            # p = u'1196258'
            if data.has_key((item_category[q], item_category[p])):
                dual_cate = (item_category[q], item_category[p])
            else:
                dual_cate = (item_category[p], item_category[q])
            # print dual_cate
            image_batch_name = generate_batch(dual_cate, data, batchsize)
            # print image_batch_name
            # print len(image_batch_name)
            image_batch_name1, image_batch_name2 = zip(*image_batch_name)
            vect_batch1 = []
            vect_batch2 = []
            for name in image_batch_name1:
                with open(vectorpath + name + '.json') as f:
                    vect_batch1.append(json.load(f))
            for name in image_batch_name2:
                with open(vectorpath + name + '.json') as f:
                    vect_batch2.append(json.load(f))
            yield image_batch_name1, cate2id[dual_cate[0]], np.array(vect_batch1).reshape(batchsize, 4096), \
                  image_batch_name2, cate2id[dual_cate[1]], np.array(vect_batch2).reshape(batchsize, 4096)

    def unsupervise_get_epoch():
        
        while True:
        # for i in range(num_batches):
            # c1 = random.choice(item_category.values())
            # c2 = random.choice(item_category.values())
            c1, c2 = random.choice(cate_2match)
            image_batch_name1 = generate_batch(c1, data, batchsize)
            image_batch_name2 = generate_batch(c2, data, batchsize)
            vect_batch1 = []
            vect_batch2 = []
            for name in image_batch_name1:
                with open(vectorpath + name + '.json') as f:
                    vect_batch1.append(json.load(f))
            for name in image_batch_name2:
                with open(vectorpath + name + '.json') as f:
                    vect_batch2.append(json.load(f))
            # print len(vect_batch1)
            # print len(vect_batch1[0])
            yield image_batch_name1, cate2id[c1], np.array(vect_batch1).reshape(batchsize, 4096), \
                  image_batch_name2, cate2id[c2], np.array(vect_batch2).reshape(batchsize, 4096)

    def vaild_get_epoch():
        def _yield_vect(image_batch_name):
            vect_batch = []
            for name in image_batch_name:
                with open(vectorpath + name + '.json') as f:
                    vect_batch.append(json.load(f))
            return np.array(vect_batch).reshape(batchsize, 4096)
        # print data.has_key(u'5')

        for cate, itemset in data.items():
            if len(itemset) / batchsize < 1:
                image_batch_name = itemset * int(batchsize / len(itemset)) + \
                                   itemset[0:batchsize % len(itemset)]
                yield image_batch_name, cate2id[cate], _yield_vect(image_batch_name)
            else:
                for i in xrange(int(len(itemset) / batchsize)):
                    image_batch_name = itemset[i * batchsize: (i+1) * batchsize]
                    yield image_batch_name, cate2id[cate], _yield_vect(image_batch_name)
                image_batch_name = itemset[-batchsize:]
                yield image_batch_name, cate2id[cate], _yield_vect(image_batch_name)

    def vaild_list_get_epoch():
        for q, p, n in data:
            cq = item_category[q]
            cp = item_category[p]
            cn = item_category[n]
            with open(vectorpath + q + '.json') as f:
                vq = np.array(json.load(f)).reshape(4096)
            with open(vectorpath + p + '.json') as f:
                vp = np.array(json.load(f)).reshape(4096)
            with open(vectorpath + n + '.json') as f:
                vn = np.array(json.load(f)).reshape(4096)
            yield (q, cate2id[cq], vq), (p, cate2id[cp], vp), (n, cate2id[cn], vn)


    if flag == "supervise":
        get_epoch = supervise_get_epoch
    elif flag == "unsupervise":
        get_epoch = unsupervise_get_epoch
    elif flag == "valid":
        get_epoch = vaild_get_epoch
    elif flag == "valid_list":
        get_epoch = vaild_list_get_epoch
    return get_epoch


def load(batch_size, test_batch_size, seed):
    # filelist = os.listdir(filepath)
    # random.shuffle(filelist)

    # valid_list = filelist[int(0.8 * len(filelist)): int(0.9 * len(filelist))]
    # test_list = filelist[int(0.9 * len(filelist)):]
    with open(filepath + "train_bycate.json", 'r') as f:
        s_train_dict = trans_into_dict(json.load(f))
    with open(filepath + "valid_bycate.json", 'r') as f:
        test_dict = dict(json.load(f))
    with open(filepath + "valid_taobao.json", 'r') as f:
        test_list = json.load(f)
    with open(filepath + "cate_item.json", 'r') as f:
        us_train_dict = dict(json.load(f))
    # with gzip.open('/Users/czy_yente/PycharmProjects/dataset/mnist.pkl.gz', 'rb') as f:
    #     train_data, dev_data, test_data = pickle.load(f)

    return (
        tianchi_generator(s_train_dict, batch_size, filepath, vectorpath, seed, 'supervise'),
        tianchi_generator(us_train_dict, batch_size, filepath, vectorpath, seed, 'unsupervise'),
        tianchi_generator(test_dict, test_batch_size, filepath, vectorpath, seed, 'valid')
        # tianchi_generator(test_list, test_batch_size, filepath, vectorpath, 'valid_list')
    )


def generate_batch(category, data, batchsize):
    # print len(data[category])
    l = int(batchsize / len(data[category]))
    return random.sample(data[category] * (l + 1), batchsize)

# def trans_into_dict(a):
#     key, value = zip(*a)
#     key = zip(*zip(*key))
#     return dict(zip(*(key, value)))


