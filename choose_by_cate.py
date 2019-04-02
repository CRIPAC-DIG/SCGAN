import json

filepath = './data/clothing/Men/'
vectorpath = '../../../../../mnt/dev_sdb1/cuizeyu/amazon_image_feature/clothing/Men/'
# filepath = './data/'
# # vectorpath = '../../../../../mnt/dev_sdb1/cuizeyu/amazon_image_feature/clothing/Men/'



with open(filepath + 'item_category.json', 'r') as f:
    item_category = dict(json.load(f))
with open(filepath + 'valid_taobao.json', 'r') as f:
    test_data = json.load(f)

with open(filepath + "cate_2match.json", 'r') as f:
    cate_2match = json.load(f)
    cate_2match = zip(*(zip(*cate_2match)))
    cate_2match = set(cate_2match)


def choose(q, p, n):
    def _cate_match(x, y):
        c_x = item_category[x]
        c_y = item_category[y]
        if (c_x, c_y) in cate_2match or (c_y, c_x) in cate_2match:
            return True
        else:
            return False

    if _cate_match(q,p) and not _cate_match(q, n):
        return 1.
    elif _cate_match(q,p) and _cate_match(q, n):
        return 0.5
    elif not _cate_match(q,p) and not _cate_match(q, n):
        print "WAAO~"
        return 0.3
    else:
        return 0.


def error_rate(q, p, r):
    def _cate_match(x, y):
        c_x = item_category[x]
        c_y = item_category[y]
        if (c_x, c_y) in cate_2match or (c_y, c_x) in cate_2match:
            return True
        else:
            return False
    if _cate_match(q,p) == r:
        return 1.
    else:
        return 0.


tt = 0.
ss = 0.

for q, p, n in test_data:
    # if choose(q, p, n) == 0.5:
    tt += choose(q, p, n)
    ss += 1.

print "auc", tt/ss

for q, p, n in test_data:
    # if choose(q, p, n) == 0.5:
    tt += error_rate(q, p, True)
    tt += error_rate(q, p, False)
    ss += 2.

print "acc", tt/ss
print 'error_rate', 1-tt/ss
