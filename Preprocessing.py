### 
import librosa
import numpy as np
import os
from pathlib import Path
import random

def _get_feature(loaded_audio_tuple, mean_signal_length: int = 100000,
                mfcc_len: int = 39):
    """
    loaded_audio_tuple: output of librosa.load(), a tuple of signal array
    and sample rate
    mean_signal_length: wanted MFCC feature length
    mfcc_len: MFCC coefficient length
    """
    signal, fs = loaded_audio_tuple
    s_len = len(signal)

    if s_len < mean_signal_length:
        pad_len = mean_signal_length - s_len
        pad_rem = pad_len % 2
        pad_len //= 2
        signal = np.pad(signal, (pad_len, pad_len + pad_rem), 'constant', 
                        constant_values = 0)
    else:
        pad_len = s_len - mean_signal_length
        pad_len //= 2
        signal = signal[pad_len:pad_len + mean_signal_length]
    mfcc = librosa.feature.mfcc(y=signal, sr=fs, n_mfcc=39)
    mfcc = mfcc.T
    feature = mfcc
    return feature


def _get_label_vector(file_name, name_to_int, num_labels): 
    """
    file_name: file name WITHOUT extension name
    name_to_int: function that converts name to int label
    num_labels: number of labels
    """
    i = name_to_int(file_name)
    v = np.zeros(num_labels)
    v[i] = 1
    return v


def shuffle_and_convert(path_func_tuples, save_path, num_labels, 
                        signal_length=100000, take=None,
                        format='wav'):
    """
    Shuffle given datasets and convert them to MFCC data.

    path_func_tuples: list of (path, func1, func2) of each dataset, where 
        path is the folder path,
        func1 is function that tells you whether to take this file in
            the first place, i.e. to filter out emotions that aren't in the 
            intersection of the datasets,
        func2 is function that converts path to an intger label
    save_path: path to which the output is saved
    num_labels: number of emotion labels
    signal_length: the target signal length achieved by padding or
        truncation
    take: number of files to be mixed in; 
        if None, mix all; 
        if one number, take that many from all datasets;
        if a list of numbers, take correspondingly
    format: audio format in file name
    """
    if not isinstance(take, list):
        take = [take] * len(path_func_tuples)

    # randomly sample from files and shuffle
    tup_subset = [] # subset of path_fun_tuples without func2
    for tup, n in zip(path_func_tuples, take):
        path, func1, func2 = tup
        samples = []
        for root, dirs, files in os.walk(path):
            for f in files:
                 if f.endswith(format) and func1(f):
                     samples.append(os.path.join(root, f))
        if isinstance(n, int):
            samples = random.sample(samples, n)
        tup_subset += [(f, func2) for f in samples]
    random.shuffle(tup_subset)

    # convert 
    xs = []
    ys = []
    for p, func in tup_subset:
        # get x
        loaded = librosa.load(p)
        x = _get_feature(loaded, mean_signal_length=signal_length)
        xs.append(x)
        # get y
        y = _get_label_vector(p, func, num_labels)
        ys.append(y)
    
    # save
    np.save(save_path, np.array({'x': np.array(xs), 'y': np.array(ys)}))
    