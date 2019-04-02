import tensorflow as tf
import json
import platform
import numpy as np
import os
import collections

"""
save the result and the model of lightrnn
includes 
         model
         best auc
"""


def wgan_saver(session, config, best_auc=0, best_epoch=0):
    saver = tf.train.Saver()
    best_auc_str = str(best_auc)
    best_auc_str = '_auc_' + best_auc_str.replace('.', '_') + '_epoch_' + str(best_epoch)
    save_name = 'model_dim' + str(config.DIM) + '_' + '.ckpt'
    saver_path = saver.save(session, "save/" + save_name)
    print("Model saved in file:", saver_path)
    # dict


def wgan_loader(session, config, path, loadwhich=0):
    saver = tf.train.Saver()
    save_name = 'model_dim' + str(config.DIM) + '_' + '.ckpt'
    saver.restore(session, path + save_name)
    print("Model loaded in file:", path + save_name)
