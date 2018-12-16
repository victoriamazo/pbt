import csv
import os
import shutil
from collections import OrderedDict
from shutil import copyfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from path import Path
from scipy.misc import imread
from wand.image import Image
from torch.autograd import Variable

from models.model_builder import Model



def idx2label(data_dir):
    dataset = data_dir.split('/')[-1]
    label_dict = {}
    if dataset == 'mnist':
        label_dict = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '0': 0}
    return label_dict


def ensure_dir(path, erase=False):
    if os.path.exists(path) and erase:
        print("Removing old folder {}".format(path))
        shutil.rmtree(path)
    if not os.path.exists(path):
        print("Creating folder {}".format(path))
        os.makedirs(path)