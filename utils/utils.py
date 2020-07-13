from __future__ import division

import os, sys, time, random
import torch
import json
import numpy as np

from config.config import RESULT_PATH, MODEL_PATH, PROJECT_PATH

def get_model_path(dataset_name, network_arch, random_seed):
    if not os.path.isdir(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    model_path = os.path.join(MODEL_PATH, "{}_{}_{}".format(dataset_name, network_arch, random_seed))
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    return model_path

def get_result_path(dataset_name, network_arch, random_seed, result_subfolder, postfix=''):
    if not os.path.isdir(RESULT_PATH):
        os.makedirs(RESULT_PATH)
    ISOTIMEFORMAT='%Y-%m-%d_%X'
    t_string = '{}'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
    result_path = os.path.join(RESULT_PATH, result_subfolder, "{}_{}_{}_{}{}".format(t_string, dataset_name, network_arch, random_seed, postfix))
    os.makedirs(result_path)
    return result_path

def time_string():
    ISOTIMEFORMAT='%Y-%m-%d %X'
    string = '[{}]'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
    return string

def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600*need_hour) / 60)
    need_secs = int(epoch_time - 3600*need_hour - 60*need_mins)
    return need_hour, need_mins, need_secs

def time_file_str():
    ISOTIMEFORMAT='%Y-%m-%d'
    string = '{}'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
    return string + '-{}'.format(random.randint(1, 10000))

def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()

def one_hot(class_labels, num_classes=None):
    if num_classes==None:
        return torch.zeros(len(class_labels), class_labels.max()+1).scatter_(1, class_labels.unsqueeze(1), 1.)
    else:
        return torch.zeros(len(class_labels), num_classes).scatter_(1, class_labels.unsqueeze(1), 1.)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def get_imagenet_dicts():
    # Imagenet class names
    idx2label = []
    cls2label = {}
    with open(os.path.join(PROJECT_PATH, "dataset_utils/imagenet_class_index.json"), "r") as read_file:
        class_idx = json.load(read_file)
        idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
        cls2label = {class_idx[str(k)][0]: class_idx[str(k)][1] for k in range(len(class_idx))}

    return idx2label, cls2label
