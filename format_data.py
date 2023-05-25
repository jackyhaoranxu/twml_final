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

import pdb
from collections import Counter, defaultdict, ChainMap

#%% this takes in raw audio files (1D), data already preprocessed

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

#%% Pads MFCC data to create uniform shapes for compatibility with TIMNET across all datasets
### Allows for arbitrary training/testing on any subset/combination of datasets

def get_uniform_mfcc(data_dir, data_names, common_labels, save_path):
    
    casia_labels = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise'] #CASIA, Chinese/German/Italian
    emodb_labels = ['angry', 'boredom', 'disgust', 'fear', 'happy', 'neutral', 'sad'] #EMODB, Chinese/German/Italian
    emovo_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'] #EMOVO, Chinese/German/Italian
    savee_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'] #SAVEE, English
    ravde_labels = ['angry', 'calm', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'] #RAVDE, English
    all_labels = [casia_labels, emodb_labels, emovo_labels, savee_labels, ravde_labels]
    # IEMOCAP_CLASS_LABELS = ("angry", "happy", "neutral", "sad") #IEMOCAP, English
    
    # get data path info
    cur_dir = os.getcwd()
    data_path = os.path.join(cur_dir, data_dir)
    
    # load specified datasets
    datasets = []
    for file in os.listdir(data_path):
        if file[:-4] in data_names:
            locals()[file[:-4].lower()] = np.load(os.path.join(data_path, file), allow_pickle=True).item()
            datasets.append(locals()[file[:-4].lower()])
    
    # get length of longest dataset to use for padding
    time_points = []
    for data in datasets:
        time_points.append(len(data['x'][1]))
        
    pad_len = np.max(time_points)
    
    padded_data = []
    
    for data in range(len(datasets)):
        # get average last row for datasets where there is no padding already
        num_repeats = pad_len - np.shape(datasets[data]['x'])[1]
        avg_pad = np.mean(datasets[data]['x'][:,-1,:], axis=0)
        padded_tr = []
        for tr in range(np.shape(datasets[data]['x'])[0]):
            c = Counter(datasets[data]['x'][tr, -1, :])
            common = c.most_common()
            if int(common[0][0]) == 0:
                # use last row for padding if mostly 0s
                pad = datasets[data]['x'][tr, -1, :]
            else:
                # otherwise use average last row
                pad = avg_pad
            padding = np.repeat(np.reshape(pad, (1,len(pad))), num_repeats, axis=0)
            padded_x = np.concatenate((datasets[data]['x'][tr, :, :], padding))
            padded_tr.append(padded_x)
        padded_mat = np.array(padded_tr)
        
        # get label indices for common labels across all datasets
        label_ind = []
        for label in range(len(all_labels[data])):
            if all_labels[data][label] in common_labels:
                label_ind.append(label)
        
        padded_dict = {'x':padded_mat, 'y':datasets[data]['y'][:,label_ind]}
        np.save(os.path.join(save_path, data_names[data]), padded_dict)
        padded_data.append(padded_dict)
    
    return datasets, padded_data

#%%

def get_train_test_data(data_dir, train_names, test_names):

    # get data path info
    cur_dir = os.getcwd()
    data_path = os.path.join(cur_dir, data_dir)
    
    # load specified datasets
    train_data = []
    test_data = []
    for file in os.listdir(data_path):
        if file[:-4] in train_names:
            locals()[file[:-4].lower()] = np.load(os.path.join(data_path, file), allow_pickle=True).item()
            train_data.append(locals()[file[:-4].lower()])
        elif file[:-4] in test_names:
            locals()[file[:-4].lower()] = np.load(os.path.join(data_path, file), allow_pickle=True).item()
            test_data.append(locals()[file[:-4].lower()])
            
    
    train_dict = {}
    for data in train_data:
        for key, value in data.items():
            if key in train_dict:
                train_dict[key] = np.concatenate((train_dict[key], value), axis=0)
            else:
                train_dict[key] = value
    
    test_dict = {}
    for data in test_data:
        for key, value in data.items():
            if key in test_dict:
                test_dict[key] = np.concatenate((test_dict[key], value), axis=0)
            else:
                test_dict[key] = value
                
    return train_dict, test_dict

#%%

orig_data_dir = 'MFCC'
data_names = ['EMODB', 'EMOVO', 'RAVDE', 'SAVEE'] # 'IEMOCAP.npy'
common_labels = ['angry', 'fear', 'happy', 'neutral', 'sad']
uni_data_dir = 'MFCC_Uniform'

# make mfcc data uniform (x/y shapes) across all datasets
orig_data, padded_data = get_uniform_mfcc(orig_data_dir, data_names, common_labels, uni_data_dir)

# ###############################################################################

# # make dataset splits
# mix_train_names = ['CASIA', 'EMODB', 'EMOVO']
# mix_test_names = ['RAVDE', 'SAVEE']
# mix_train, mix_test = get_train_test_data(uni_data_dir, mix_train_names, mix_test_names)

# eng_train_names = ['RAVDE', 'SAVEE']
# eng_test_names = ['CASIA', 'EMODB', 'EMOVO']
# eng_train, eng_test = get_train_test_data(uni_data_dir, eng_train_names, eng_test_names)

# one_vs_all_train1_names = ['CASIA']
# one_vs_all_test1_names = ['EMODB', 'EMOVO', 'RAVDE', 'SAVEE']
# one_vs_all_train1, one_vs_all_test1 = get_train_test_data(uni_data_dir, one_vs_all_train1_names, one_vs_all_test1_names)

# one_vs_all_train2_names = ['EMODB']
# one_vs_all_test2_names = ['CASIA', 'EMOVO', 'RAVDE', 'SAVEE']
# one_vs_all_train2, one_vs_all_test2 = get_train_test_data(uni_data_dir, one_vs_all_train2_names, one_vs_all_test2_names)




    














