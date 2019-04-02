"""
find which config to use and select

"""
import json

def Config(tianchivect, flag=None):
    if flag == None:
        return config_init(tianchivect)
    elif flag == "haha":
        return haha()
    else:
        raise ValueError("Invalid model: %s", flag)


class config_init(object):
    epoch_num = 30000

    max_grad_norm = 1
    init_scale = 0.05
    learning_rate = 0.001  # 0.2  0.01

    adagrad_eps = 1e-5
    DIM = 128  # Model dimensionality
    OUTPUT_DIM = 100
    D_hidden_layer = 1  # 10
    disc_iter = 5
    gen_iter = 1
    # category_num = 71
    BATCH_SIZE = 30  # Batch size # set it to 2 and the normalization could be in easy
    CRITIC_ITERS = 5  # For WGAN and WGAN-GP, number of critic iters per gen iter
    LAMBDA = 10  # Gradient penalty lambda hyperparameter
    delta = 0.01 # the diversity
    ITERS = 1000  # How many generator iterations to train for
    GIMG = 4096
    ratio_super = (0.5, 0.5) # supervise : unsupervise
    gpuid = 1

    def __init__(self, tianchivect):
        # self.vocab_size = len(reader.vocab.words) # 10000
        print "config~"
        print "learning_rate", self.learning_rate
        print "max_grad_norm", self.max_grad_norm
        print "disc_iter", self.disc_iter
        print "gen_iter", self.gen_iter
        print "DIM", self.DIM
        print "delta", self.delta
        print "ratio_super", self.ratio_super
        with open(tianchivect.filepath + "cate_2match.json", 'r') as f:
            cate_2match = json.load(f)
        self.cat_match = len(cate_2match)
        self.catmatch_id = cate_2match
        self.cate2id = tianchivect.cate2id
        self.category_num = len(tianchivect.cate2id)
        print tianchivect.filepath
        print "category_num", self.category_num
        print "cat_match_num", self.cat_match

class haha(object):
    def __init__(self, loader):
        print "haha~"

