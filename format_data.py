# -*- coding: utf-8 -*-
"""
Created on Mon May 22 12:54:42 2023

@author: Drew
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from Model import TIMNET_Model
import argparse

### MY MODIFICATION
from Model import my_test, create_timnet_model
from pathlib import Path
import librosa

#%%

def get_feature(file_path: str, mfcc_len: int = 39, mean_signal_length: int = 100000):
    """
    file_path: Speech signal folder
    mfcc_len: MFCC coefficient length
    mean_signal_length: MFCC feature average length
	"""
    
    signal, fs = librosa.load(file_path)
    s_len = len(signal)
    
    if s_len < mean_signal_length:
        pad_len = mean_signal_length - s_len
        pad_rem = pad_len % 2
        pad_len //= 2
        signal = np.pad(signal, (pad_len, pad_len + pad_rem), 'constant', constant_values = 0)
    else:
        pad_len = s_len - mean_signal_length
        pad_len //= 2
        signal = signal[pad_len:pad_len + mean_signal_length]
    mfcc = librosa.feature.mfcc(y=signal, sr=fs, n_mfcc=39)
    mfcc = mfcc.T
    feature = mfcc
    return feature

#%%

cur_dir = os.getcwd()
data_dir = 'MFCC'

casia = np.load(os.path.join(cur_dir, data_dir, 'CASIA.npy'), allow_pickle=True).item()
emodb = np.load(os.path.join(cur_dir, data_dir, 'EMODB.npy'), allow_pickle=True).item()
emovo = np.load(os.path.join(cur_dir, data_dir, 'EMOVO.npy'), allow_pickle=True).item()
iemocap = np.load(os.path.join(cur_dir, data_dir, 'IEMOCAP.npy'), allow_pickle=True).item()
ravde = np.load(os.path.join(cur_dir, data_dir, 'RAVDE.npy'), allow_pickle=True).item()
savee = np.load(os.path.join(cur_dir, data_dir, 'SAVEE.npy'), allow_pickle=True).item()



# this takes in raw audio files (1D), data already preprocessed
emodb_format = get_feature(emodb['x'], 39, 100000)




















