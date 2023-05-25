# -*- coding: utf-8 -*-
"""
Created on Thu May 25 01:41:48 2023

@author: Drew
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt


#%% imports csv files
def import_csv(filename):
    '''Imports csv into a simple structure.

    Arguments:
        filename {str} -- filename to import.
    Returns a tuple of:
        column_names {list of M str} -- list of all column names.
        values {list [M][N] of float if possible, str otherwise} -- list of all column values. First
            index corresponds to column number.
    '''
    with open(filename, 'r') as f:
        rdr = csv.reader(f)

        line = next(rdr)
        column_names = [i.strip() for i in line]

        values = [[] for _ in column_names]

        for li in rdr:
            for idof, vdof in enumerate(li):
                try:
                    v = float(vdof)
                except ValueError:
                    v = vdof
                values[idof].append(v)
    return column_names, values

#%% set up path info and load data metrics

data_types = ['audio', 'mfcc']
metric_types = ['CONF_MATRIX', 'EVAL_METRICS']
results_dir = 'Results'

# get results path info
cur_dir = os.getcwd()
results_path = os.path.join(cur_dir, results_dir)

# load specified metrics
conf_mat_audio = []
eval_met_audio = []
conf_mat_mfcc = []
eval_met_mfcc = []
for file in os.listdir(results_path):
    f = file.split('_')
    metric = '_'.join(f[0:2])
    train_set = f[2]
    test_set = f[3]
    dtype = f[4][:-4]
    if metric == 'CONF_MATRIX' or metric == 'EVAL_METRICS':
        if metric == 'CONF_MATRIX':
            if dtype == 'audio':
                locals()['_'.join([metric, train_set, test_set, dtype, 'cols']).lower()],\
                locals()['_'.join([metric, train_set, test_set, dtype, 'vals']).lower()]\
                = import_csv(os.path.join(results_path, file))            
                conf_mat_audio.append(locals()['_'.join([metric, train_set, test_set, dtype, 'vals']).lower()])
            elif dtype == 'mfcc':
                locals()['_'.join([metric, train_set, test_set, dtype, 'cols']).lower()],\
                locals()['_'.join([metric, train_set, test_set, dtype, 'vals']).lower()]\
                = import_csv(os.path.join(results_path, file))            
                conf_mat_mfcc.append(locals()['_'.join([metric, train_set, test_set, dtype, 'vals']).lower()])
        elif metric =='EVAL_METRICS':
            if dtype == 'audio':
                locals()['_'.join([metric, train_set, test_set, dtype, 'cols']).lower()],\
                locals()['_'.join([metric, train_set, test_set, dtype, 'vals']).lower()]\
                = import_csv(os.path.join(results_path, file))            
                eval_met_audio.append(locals()['_'.join([metric, train_set, test_set, dtype, 'vals']).lower()])
            elif dtype == 'mfcc':
                locals()['_'.join([metric, train_set, test_set, dtype, 'cols']).lower()],\
                locals()['_'.join([metric, train_set, test_set, dtype, 'vals']).lower()]\
                = import_csv(os.path.join(results_path, file))            
                eval_met_mfcc.append(locals()['_'.join([metric, train_set, test_set, dtype, 'vals']).lower()])
        

        
        
        
        
        
        
        
        
        
        
        
        
        