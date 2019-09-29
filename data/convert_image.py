import numpy as np
import tensorflow as tf
import os
gpuid = 5
# os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(gpuid)
import vgg16
import utils
import collections
import json


batchsize = 100

pathfile = "../../../../../mnt/dev_sdb1/cuizeyu/tianchi_fm_img/"
os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(gpuid)
init_data_path = "../../../../../mnt/dev_sdb1/cuizeyu/image_vector/"

# fid = open('item_img.txt', 'w')


session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
session_config.gpu_options.allow_growth = True

# Train loop

# def image_process(batch, batchsize, gpuid):
    # with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
with tf.device('/gpu:' + str(gpuid)):
    graph = tf.Graph()
    with graph.as_default():
        vgg = vgg16.Vgg16()
    with tf.Session(graph=graph, config=session_config) as sess:
        images = tf.placeholder("float", [batchsize, 224, 224, 3])

        with tf.name_scope("content_vgg"):
            vgg.build(images)
        item = os.listdir(init_data_path)
        def _convertname(item):
            return item[0:-5] + '.jpg'
        item = map(_convertname, item)
        itemset = set(item)
        filelist = os.listdir(pathfile)
        fileset =set(filelist)
        filelist = list(fileset - itemset)
        print 'all', len(fileset)
        print 'already', len(itemset)
        print 'rest', len(filelist)
        # item_img = collections.OrderedDict()
        batch = []
        for idx, image in enumerate(filelist):
            if len(batch) % batchsize == 0:
                batch = np.asarray(utils.load_image(pathfile + image).reshape((1, 224, 224, 3)))
                # print batch.shape
                image_name = [image[0:-4]]
            else:
                temp = utils.load_image(pathfile + image).reshape((1, 224, 224, 3))
                image_name.append(image[0:-4])
                batch = np.concatenate((batch, temp), 0)
                # print batch.shape
                if len(batch) % batchsize == 0:
                    feed_dict = {images: batch}
                    image_vector = sess.run(vgg.relu7, feed_dict=feed_dict)
                    for jdx, name in enumerate(image_name):
                        # item_img[name] = map(float, list(image_vector[jdx]))
                        pp = True
                        while pp:
                            try:
                                with open(init_data_path + str(name) + '.json', 'w') as fid:
                                    fid.write(json.dumps(map(float, list(image_vector[jdx]))))
                                    pp = False
                            except:
                                continue
            if idx % (batchsize * 10) == 0:
                print idx

# fid.close()
# with open('item_img.json', 'w') as f:
#     f.write(json.dumps(item_img.items()))

