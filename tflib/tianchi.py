import numpy as np
import random
import os
import skimage
import skimage.io
import skimage.transform
import collections
import json
import urllib


def load_image(path, ROW, COL, LAY):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to ROW, COL
    resized_img = skimage.transform.resize(crop_img, (ROW, COL))
    return resized_img


def tianchi_generator(data, batchsize, filepath, vectorpath, ROW, COL, LAY):
    length = len(data)
    num_batches = length / batchsize

    def get_epoch():
        for i in range(num_batches):
            image_batch_name = data[i * batchsize: i * batchsize + batchsize]
            # print len(image_batch_name)
            image_batch = []
            vect_batch = []
            for name in image_batch_name:
                image_batch.append(load_image(filepath + name, ROW, COL, LAY))
                with open(vectorpath + name[0:-4] + '.json') as f:
                    vect_batch.append(json.load(f))

            yield (np.array(image_batch).reshape(batchsize, ROW * COL * LAY),
            np.array(vect_batch).reshape(batchsize, 4096))

    return get_epoch


def load(batch_size, test_batch_size, ROW, COL, LAY, GIMG):
    filepath = '../../../../../mnt/dev_sdb1/cuizeyu/tianchi_fm_img/'
    # filepath = '/Users/czy_yente/Downloads/ali_data/item/'
    vectorpath = '../../../../../mnt/dev_sdb1/cuizeyu/image_vector/'
    filelist = os.listdir(filepath)
    random.shuffle(filelist)

    train_list = filelist[0: int(0.8 * len(filelist))]
    valid_list = filelist[int(0.8 * len(filelist)): int(0.9 * len(filelist))]
    test_list = filelist[int(0.9 * len(filelist)):]

    # with gzip.open('/Users/czy_yente/PycharmProjects/dataset/mnist.pkl.gz', 'rb') as f:
    #     train_data, dev_data, test_data = pickle.load(f)

    return (
        tianchi_generator(train_list, batch_size, filepath, vectorpath, ROW, COL, LAY),
        tianchi_generator(valid_list, test_batch_size, filepath, vectorpath, ROW, COL, LAY),
        tianchi_generator(test_list, test_batch_size, filepath, vectorpath, ROW, COL, LAY),
        tianchi_generator(train_list, GIMG, filepath, vectorpath, ROW, COL, LAY)
    )

