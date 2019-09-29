#!/usr/bin/python -u
import collections
import json
init_data_path = "E:\\down\\data\\clothingmatching\\"
import itertools
import time
import os


def preprosess_item():
    """
    itemset
    """
    def _findname(xx):
        return xx[0:-4]
    image_set = set(map( _findname, os.listdir(init_data_path + 'tianchi_fm_img\\')))
    # with open('cate_2match.json', 'r') as f:
    #     cate_set = set(itertools.chain.from_iterable(json.load(f)))

    itemfile = 'dim_items(new).txt'
    item_category = collections.OrderedDict()
    with open(init_data_path + itemfile, 'r') as f:
        for line in f.readlines():
            temp = line.strip().split(' ')
            if temp[0] not in image_set:
                continue
            # if temp[1] not in cate_set:
            #     continue
            item_category[temp[0]] = temp[1]

    with open('item_category.json', 'w') as f:
        f.write(json.dumps(item_category.items()))

    print 'itemnum', len(set(item_category.keys()))
    print 'catnum', len(set(item_category.values()))


def preprosess_matchset():
    matchfile = "dim_fashion_matchsets(new).txt"
    fashion_match = []
    with open('item_category.json', 'r') as f:
        item_category = collections.OrderedDict(json.load(f))

    with open(init_data_path + matchfile, 'r') as f:
        for line in f.readlines():
            temp = line.strip().split()
            temp = temp[1].strip().split(';')
            fashion_match.append([])
            for temp1 in temp:
                fashion_match[-1].append(filter(lambda x: item_category.has_key(x),
                                                temp1.strip().split(',')))
    for idx, line in enumerate(fashion_match):
        jdx = 0
        while jdx < len(line):
            if line[jdx] == []:
                del fashion_match[idx][jdx]
            else:
                jdx += 1

    with open('fashion_match.json', 'w') as f:
        f.write(json.dumps(fashion_match))


def cal_item_match():
    with open('item_category.json', 'r') as f:
        item_category = collections.OrderedDict(json.load(f))
    with open('fashion_match.json', 'r') as f:
        fashion_match = json.load(f)
    item2itemdict = collections.OrderedDict()
    for idx, match in enumerate(fashion_match):
        for jdx, altertive in enumerate(match):
            for alt in altertive:
                if item2itemdict.has_key(alt):
                    item2itemdict[alt] += list(itertools.chain.from_iterable(match[0:jdx] + match[jdx+1:]))
                else:
                    item2itemdict[alt] = list(itertools.chain.from_iterable(match[0:jdx] + match[jdx+1:]))
    print len(item2itemdict)

    for item in item2itemdict.keys():
        if not len(item2itemdict[item]) == len(set(item2itemdict[item])):
            item2itemdict[item] = list(set(item2itemdict[item]))

    with open('item_match.json', 'w') as f:
        f.write(json.dumps(item2itemdict.items()))


def cal_category_match():
    with open('item_category.json', 'r') as f:
        item_category = collections.OrderedDict(json.load(f))
    with open('fashion_match.json', 'r') as f:
        fashion_match = json.load(f)
    print len(fashion_match)
    t = 0
    cate_match = []
    for idx, match in enumerate(fashion_match):
        cate_match.append([])
        for altertive in match:
            if not item_category.has_key(altertive[0]):
                t += 1
                continue
            # cate = item_category[altertive[0]]
            cate_match[-1].append(item_category[altertive[0]])
        if idx == 9063 or idx == 17576:
            for altertive in match:
                if not item_category.has_key(altertive[0]):
                    t += 1
                    continue
                cate = item_category[altertive[0]]
                for jdx, alter in enumerate(altertive):
                    if not item_category.has_key(alter):
                        t += 1
                        continue
                    if cate != item_category[alter]:
                        cate_match.append(cate_match[-1][:])
                        cate_match[-1][jdx] = item_category[alter]
    print len(cate_match)
    print t

    cate_2match = set([])
    for line in cate_match:
        for idx in range(len(line) - 1):
            for jdx in range(idx + 1, len(line)):
                if not ((line[idx], line[jdx]) in cate_match or (line[jdx], line[idx]) in cate_match):
                    cate_2match.add((line[idx], line[jdx]))

    print len(cate_2match)
    with open('cate_2match.json', 'w') as f:
        f.write(json.dumps(list(cate_2match)))


def cal_temp():
    with open('fashion_match.json', 'r') as f:
        fashion_match = json.load(f)
    # print list(itertools.chain.from_iterable(fashion_match))
    iteminmatch = set(list(itertools.chain.from_iterable(itertools.chain.from_iterable(fashion_match))))



def watch_itemimg():
    import os
    def _convertname(item):
        return item[0:-5]
    ttt = time.time()
    # item_img = collections.OrderedDict()
    f = open('item_img.txt', 'r')
    t = 0
    lines = f.readlines()
    f.close()
    item = os.listdir(init_data_path)
    item = map(_convertname, item)
    print item[0:100]
    itemset = set(item)
    while len(itemset) < len(lines):
        try:
            for line in lines:
                temp = line.strip().split(',')
                name = temp[0]
                t += 1
                if name in itemset:
                    continue
                itemset.add(name)
                vector = map(float, temp[1:])
                with open(init_data_path + str(name) + '.json', 'w') as fid:
                    fid.write(json.dumps(vector))
                print t
                if t % 100 == 0:
                    print t

        except:
            continue
    print 'times:', time.time() - ttt


def temp():
    with open('item_category.json', 'r') as f:
        item_category = collections.OrderedDict(json.load(f))
    itembycat = collections.OrderedDict()
    for item, cat in item_category.items():
        if not itembycat.has_key(cat):
            itembycat[cat] = [item]
        else:
            itembycat[cat].append(item)
    print len(itembycat)
    with open('itembycat.json', 'w') as f:
        f.write(json.dumps(itembycat.items()))
    cate_list = sorted(itembycat.keys(), key=lambda x: len(itembycat[x]))
    for cat in cate_list:
        print cat, len(itembycat[cat])


def cate_item():
    with open("item_category.json", 'r') as f:
        item_category = json.load(f)
    category_item = dict()
    for item, cate in item_category.items():
        if category_item.has_key(cate):
            category_item[cate].append(item)
        else:
            category_item[cate] = [item]

    with open("cate_item.json", 'w') as f:
        f.write(json.dumps(category_item))


if __name__ == '__main__':
    # preprosess_item()
    # preprosess_matchset()
    # cal_category_match()
    # cal_item_match()
    # watch_itemimg()
    # temp()
    # cate_item()
    