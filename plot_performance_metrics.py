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


#%% plot confusion matrices

def plot_conf_mat(audio, mfcc, axis_labels, data_sets, figure_path):
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(audio)
    plt.colorbar(fraction=0.05)
    plt.xticks(ticks=np.arange(0,len(axis_labels)), labels=axis_labels)
    plt.yticks(ticks=np.arange(0,len(axis_labels)), labels=axis_labels)
    plt.title('Audio Padding')
    
    plt.subplot(1,2,2)
    plt.imshow(mfcc)
    plt.colorbar(fraction=0.05)
    plt.xticks(ticks=np.arange(0,len(axis_labels)), labels=axis_labels)
    plt.yticks(ticks=np.arange(0,len(axis_labels)), labels=axis_labels)
    plt.title('MFCC Padding')
    
    plt.suptitle(data_sets, y=0.82, fontweight='bold')
    
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.6,
                        hspace=0.4)
    plt.savefig(os.path.join(figure_path, '{}.png'.format(data_sets)),
                dpi=300, bbox_inches='tight')
    plt.close()
    

#%% plot model accuracy

def plot_accuracy(audio, mfcc, test_dataset, title, figure_path, save_name):
    plt.figure()
    plt.plot(audio, linewidth=2, color='C0')
    plt.plot(mfcc, linewidth=2, color='C1')
    plt.xticks(ticks=np.arange(0,len(test_dataset)), labels=test_dataset)
    plt.xlabel('Test Dataset')
    plt.ylabel('Accuracy (%)')
    plt.legend(['Audio Padding', 'MFCC Padding'])
    plt.title(title, fontweight='bold')
    plt.savefig(os.path.join(figure_path, '{}.png'.format(save_name)),
                dpi=300, bbox_inches='tight')
    plt.close()

#%% set up path info and load data metrics

data_types = ['audio', 'mfcc']
metric_types = ['CONF_MATRIX', 'EVAL_METRICS']
results_dir = 'Results'

# get results path info
cur_dir = os.getcwd()
results_path = os.path.join(cur_dir, results_dir)

# load specified metrics
for file in os.listdir(results_path):
    f = file.split('_')
    metric = '_'.join(f[0:2])
    train_set = f[2]
    test_set = f[3]
    dtype = f[4][:-4]
    if metric == 'CONF_MATRIX' or metric == 'EVAL_METRICS':
        if metric == 'CONF_MATRIX':
            if dtype == 'audio':
                cols_cm, locals()['_'.join([metric, train_set, test_set, dtype, 'vals']).lower().replace('-', '')]\
                = import_csv(os.path.join(results_path, file))            
                
                locals()['_'.join([metric, train_set, test_set, dtype, 'vals']).lower().replace('-', '')]\
                    = np.array(locals()['_'.join([metric, train_set, test_set, dtype, 'vals']).lower().replace('-', '')][1:]).T / \
                        np.sum(np.array(locals()['_'.join([metric, train_set, test_set, dtype, 'vals']).lower().replace('-', '')][1:]).T)
            elif dtype == 'mfcc':
                cols_cm, locals()['_'.join([metric, train_set, test_set, dtype, 'vals']).lower().replace('-', '')]\
                = import_csv(os.path.join(results_path, file))            
                
                locals()['_'.join([metric, train_set, test_set, dtype, 'vals']).lower().replace('-', '')]\
                    = np.array(locals()['_'.join([metric, train_set, test_set, dtype, 'vals']).lower().replace('-', '')][1:]).T / \
                        np.sum(np.array(locals()['_'.join([metric, train_set, test_set, dtype, 'vals']).lower().replace('-', '')][1:]).T)
        elif metric =='EVAL_METRICS':
            if dtype == 'audio':
                cols_em, locals()['_'.join([metric, train_set, test_set, dtype, 'vals']).lower().replace('-', '')]\
                = import_csv(os.path.join(results_path, file))            
                
                locals()['_'.join([metric, train_set, test_set, dtype, 'vals']).lower().replace('-', '')]\
                    = np.array(locals()['_'.join([metric, train_set, test_set, dtype, 'vals']).lower().replace('-', '')][1:]).T
                    
                locals()['_'.join(['acc', train_set, test_set, dtype, 'vals']).lower().replace('-', '')] = \
                    locals()['_'.join([metric, train_set, test_set, dtype, 'vals']).lower().replace('-', '')][5,0]
            elif dtype == 'mfcc':
                cols_em, locals()['_'.join([metric, train_set, test_set, dtype, 'vals']).lower().replace('-', '')]\
                = import_csv(os.path.join(results_path, file))            
                
                locals()['_'.join([metric, train_set, test_set, dtype, 'vals']).lower().replace('-', '')]\
                    = np.array(locals()['_'.join([metric, train_set, test_set, dtype, 'vals']).lower().replace('-', '')][1:]).T
                    
                locals()['_'.join(['acc', train_set, test_set, dtype, 'vals']).lower().replace('-', '')] = \
                    locals()['_'.join([metric, train_set, test_set, dtype, 'vals']).lower().replace('-', '')][5,0]
        

#%% generate confusion matrice plots

figures_dir = 'Figures'
conf_mat_dir = 'ConfMat'
conf_mat_path = os.path.join(cur_dir, figures_dir, conf_mat_dir)

# train 3, test 1
plot_conf_mat(conf_matrix_emodbemovoravde_savee_audio_vals, conf_matrix_emodbemovoravde_savee_mfcc_vals,\
              cols_cm[1:], 'EMODB-EMOVO-RAVDE_SAVEE', conf_mat_path)
    
plot_conf_mat(conf_matrix_emodbemovosavee_ravde_audio_vals, conf_matrix_emodbemovosavee_ravde_mfcc_vals,\
              cols_cm[1:], 'EMODB-EMOVO-SAVEE_RAVDE', conf_mat_path)
    
plot_conf_mat(conf_matrix_emodbravdesavee_emovo_audio_vals, conf_matrix_emodbravdesavee_emovo_mfcc_vals,\
              cols_cm[1:], 'EMODB-RAVDE-SAVEE_EMOVO', conf_mat_path)
    
plot_conf_mat(conf_matrix_emovoravdesavee_emodb_audio_vals, conf_matrix_emovoravdesavee_emodb_mfcc_vals,\
              cols_cm[1:], 'EMOVO-RAVDE-SAVEE_EMODB', conf_mat_path)

# language split
plot_conf_mat(conf_matrix_emodbemovo_ravdesavee_audio_vals, conf_matrix_emodbemovo_ravdesavee_mfcc_vals,\
              cols_cm[1:], 'EMODB-EMOVO_RAVDE-SAVEE', conf_mat_path)
    
plot_conf_mat(conf_matrix_ravdesavee_emodbemovo_audio_vals, conf_matrix_ravdesavee_emodbemovo_mfcc_vals,\
              cols_cm[1:], 'RAVDE-SAVEE_EMODB-EMOVO', conf_mat_path)    
        
        
#%% generate accuracy plots

acc_dir = 'Accuracy'
acc_path = os.path.join(cur_dir, figures_dir, acc_dir)

# train 3, test 1
train3_test1_acc_audio = [acc_emodbemovoravde_savee_audio_vals*100,\
                          acc_emodbemovosavee_ravde_audio_vals*100,\
                          acc_emodbravdesavee_emovo_audio_vals*100,\
                          acc_emovoravdesavee_emodb_audio_vals*100]
train3_test1_acc_mfcc = [acc_emodbemovoravde_savee_mfcc_vals*100,\
                          acc_emodbemovosavee_ravde_mfcc_vals*100,\
                          acc_emodbravdesavee_emovo_mfcc_vals*100,\
                          acc_emovoravdesavee_emodb_mfcc_vals*100]
    
train3_test1_title = 'Train on 3 Datasets, Test on 1 Dataset'
train3_test1_test_datasets = ['SAVEE', 'RAVDE', 'EMOVO', 'EMODB']

plot_accuracy(train3_test1_acc_audio, train3_test1_acc_mfcc, train3_test1_test_datasets, train3_test1_title, acc_path, 'Train3Test1')



# language split
lang_acc_audio = [acc_emodbemovo_ravdesavee_audio_vals*100,\
                    acc_ravdesavee_emodbemovo_audio_vals*100]
lang_acc_mfcc = [acc_emodbemovo_ravdesavee_mfcc_vals*100,\
                    acc_ravdesavee_emodbemovo_mfcc_vals*100]
    
lang_title = 'Train/Test on Different Language Datasets'
lang_test_datasets = ['SAVEE-RAVDE (Chinese/German/Italian)', 'EMOVO-EMODB (English)']

plot_accuracy(lang_acc_audio, lang_acc_mfcc, lang_test_datasets, lang_title, acc_path, 'Language')
        
        
        
        
        
        
