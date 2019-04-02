import os
import sys
import time
import numpy
from  datetime  import  *
import json
from sklearn import decomposition


filepath_test = './data/test_taobao.json'
filepath_feature = '/home/cuizeyu/image_vector/'
filepath = './data/'

print('now():' + str(datetime.now()))
with open(filepath + "cate_2match.json", 'r') as f:
    cate_2match = json.load(f)

print('now():' + str(datetime.now()))
with open(filepath + "item_category.json", 'r') as f:
    item_category = json.load(f)

print('loading data')
print('now():' + str(datetime.now()))
with open(filepath_test, 'r') as f:
    test_taobao = json.load(f)
    test_set_size = len(test_taobao)
    print('test_data size:%d' % test_set_size)

fi = []
fj = []
fk = []

for a in range(test_set_size):
    set = test_taobao[a]
    i = set[0]
    j = set[1]
    k = set[2]
    with open(filepath_feature + i + '.json', 'r') as f:
        fi.append(numpy.asarray(json.load(f)))
    with open(filepath_feature + j + '.json', 'r') as f:
        fj.append(numpy.asarray(json.load(f)))
    with open(filepath_feature + k + '.json', 'r') as f:
        fk.append(numpy.asarray(json.load(f)))


best = 0.0

for n in range(0, 6):
    pca_n = 2 * (4 ** n)
    if pca_n != 2:
        continue
    pca = decomposition.PCA(n_components=pca_n, copy=True)
    vi = pca.fit_transform(fi)
    vj = pca.fit_transform(fj)
    vk = pca.fit_transform(fk)
    '''
    ij = numpy.mean((vi - vj) ** 2, axis=1)
    ik = numpy.mean((vi - vk) ** 2, axis=1)
    '''
    ij = numpy.dot(vi, numpy.transpose(vj)).diagonal()
    ik = numpy.dot(vi, numpy.transpose(vk)).diagonal()

    count = 0.
    for i in range(test_set_size):
        if ([item_category[test_taobao[i][0]], item_category[test_taobao[i][2]]] in cate_2match
            or [item_category[test_taobao[i][2]],
                item_category[test_taobao[i][0]]] in cate_2match):
            if (ij[i] > ik[i]):
                count = count + 1
        else:
            count = count + 1

    performance = float(count / test_set_size)
    print (pca_n, performance)

    if performance > best:
        best = performance

print ('best performance is %f' % best)






