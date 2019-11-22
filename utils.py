import os
import cv2
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import sys
import pickle
from datetime import datetime

# DATA_DIR = '/data/'
PROJECT_DIR = '/home/kaandonbekci/Projects/Projects_2/recursion'
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
RECURSION_DIR = os.path.join(DATA_DIR, 'recursion')
RECURSION_TRAIN_DIR = os.path.join(RECURSION_DIR, 'train')
RECURSION_TEST_DIR = os.path.join(RECURSION_DIR, 'test')
DUMP_DIR = os.path.join(PROJECT_DIR, 'dumps')
LOG_DIR = os.path.join(PROJECT_DIR, 'logs')
CELL_TYPES = ['HEPG2', 'HUVEC', 'RPE', 'U2OS']
PLATES = ['Plate1', 'Plate2', 'Plate3', 'Plate4']
LETTER_TO_IX = {}
for ix, letter in enumerate(string.ascii_uppercase[1:15]):
    LETTER_TO_IX[letter] = ix 
IX_TO_LETTER = {v: k for k, v in LETTER_TO_IX.items()}

def parse_filename(s, full_path=False):
    ''' Returns row, col, site, channel of a string in the format of the kaggle filename. '''
    #first _ is always 3rd index
    if full_path:
        s = s[-13:]
    col = LETTER_TO_IX[s[0]]
    row = int(s[1:3]) - 2
    site = int(s[5:6]) - 1
    channel = int(s[8:9]) - 1
    return row, col, site, channel  

def get_image_path(experiment, plate, well, site, channel, train):
    path = os.path.join(RECURSION_TRAIN_DIR if train else RECURSION_TEST_DIR, f'{experiment}/Plate{plate}/{well}_s{site}_w{channel}.png')
    return path

def read_image(path, as_float=True):
    if as_float:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0
    else: 
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img

def read_parse_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0
    info = parse_filename(path, True)
    return img, info

def plot_cell(sample_img):    
    channels = ['Nuclei', 'Endoplasmic reticuli', 'Actin', 'Nucleoli', 'Mitochondria', 'Golgi apparatus']
    cmaps = ['gist_ncar','terrain', 'gnuplot' ,'rainbow','PiYG', 'gist_earth']

    fig=plt.figure(figsize=(20, 15))
    for i in range(1,6+1):
        fig.add_subplot(1, 6, i)
        plt.imshow(sample_img[i-1, :, :,],cmap=cmaps[i-1]);
        plt.axis('off');
        plt.title(f'{channels[i-1]}')
    fig.suptitle("Single image channels", y=0.65, fontsize=15)
    plt.show()

def tf_fix(tf):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)