"""
@author: Jiaxin Ye
@contact: jiaxin-ye@foxmail.com

Utilities modified by Jacky Xu @ May 2023 for a course project.
"""

# -*- coding:UTF-8 -*-
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from Model import TIMNET_Model
import argparse

### MY MODIFICATION
from Model import my_test
from pathlib import Path
from format_data import get_train_test_data

parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default="train")
parser.add_argument('--model_path', type=str, default='./Models/')
parser.add_argument('--result_path', type=str, default='./Results/')
parser.add_argument('--test_path', type=str, default='./Test_Models/EMODB_46')
# FORMAT: TrainData1-TrainData2_TestData1-TestData2
parser.add_argument('--data', type=str, default='EMODB')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--beta1', type=float, default=0.93)
parser.add_argument('--beta2', type=float, default=0.98)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epoch', type=int, default=60)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--random_seed', type=int, default=46)
parser.add_argument('--activation', type=str, default='relu')
parser.add_argument('--filter_size', type=int, default=39)
parser.add_argument('--dilation_size', type=int, default=8)# If you want to train model on IEMOCAP, you should modify this parameter to 10 due to the long duration of speech signals.
parser.add_argument('--kernel_size', type=int, default=2)
parser.add_argument('--stack_size', type=int, default=1)
parser.add_argument('--split_fold', type=int, default=10)
parser.add_argument('--gpu', type=str, default='0')

### MY MODIFICATION:
### Add parser arguments that allow custom dataset input
parser.add_argument('--data_path', type=str, default=None)

# EXAMPLE: --class_labels label1 label2 label3 ...
parser.add_argument('--class_labels', type=str, nargs='+', default=None)

### Add parser arguments that specify weights path (the original code
### makes this via set string rules, consequently allowing only the 
### weights trained from the six defualt datasets.)
parser.add_argument('--weights_path', type=str, default=None)

### ADDITIONAL ARGUMENTS FOR TRAIN/TEST ON GROUPED DATASETS
parser.add_argument('--train_datasets', type=str, nargs='+', default=None)
parser.add_argument('--test_datasets', type=str, nargs='+', default=None)
parser.add_argument('--data_type', type=str, default=None)

args = parser.parse_args()

if args.data=="IEMOCAP" and args.dilation_size!=10:
    args.dilation_size = 10
    
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth=True 
# session = tf.compat.v1.Session(config=config)
# print(f"###gpus:{gpus}")

CLASS_LABELS_finetune = ("angry", "fear", "happy", "neutral", "sad")
CASIA_CLASS_LABELS = ("angry", "fear", "happy", "neutral", "sad", "surprise") #CASIA, Chinese/German/Italian
EMODB_CLASS_LABELS = ("angry", "boredom", "disgust", "fear", "happy", "neutral", "sad") #EMODB, Chinese/German/Italian
SAVEE_CLASS_LABELS = ("angry","disgust", "fear", "happy", "neutral", "sad", "surprise") #SAVEE, English
RAVDE_CLASS_LABELS = ("angry", "calm", "disgust", "fear", "happy", "neutral","sad","surprise") #RAVDE, English
IEMOCAP_CLASS_LABELS = ("angry", "happy", "neutral", "sad") #IEMOCAP, English
EMOVO_CLASS_LABELS = ("angry", "disgust", "fear", "happy","neutral","sad","surprise") #EMOVO, Chinese/German/Italian
CLASS_LABELS_dict = {"CASIA": CASIA_CLASS_LABELS,
               "EMODB": EMODB_CLASS_LABELS,
               "EMOVO": EMOVO_CLASS_LABELS,
               "IEMOCAP": IEMOCAP_CLASS_LABELS,
               "RAVDE": RAVDE_CLASS_LABELS,
               "SAVEE": SAVEE_CLASS_LABELS}

### MY MODIFICATION:
    
common_labels = ['angry', 'fear', 'happy', 'neutral', 'sad']


# use data_type argument to specify which dataset to use for model    
if args.data_type == 'mfcc':
    data_dir = 'MFCC_Uniform' # padded MFCC data
elif args.data_type == 'audio':
    data_dir = 'uniform_by_audio' # padded raw audio data

# parse datasets to use for training/testing
train_names = args.train_datasets
test_names = args.test_datasets

# combine datasets for training/testing
train_data, test_data = get_train_test_data(data_dir, train_names, test_names)

# train model
model_init = TIMNET_Model(args=args, input_shape=train_data['x'].shape[1:], class_label=common_labels)
model_trained = model_init.my_train(train_data['x'], train_data['y'])

# test model
my_test(args=args, model=model_trained, x=test_data['x'], y=test_data['y'], class_labels=common_labels, dataset_name=data_dir)
    

### Either load data from path, or load by default style as written originally.
# if args.data_path is None:
#     data = np.load("./MFCC/"+args.data+".npy",allow_pickle=True).item()
#     CLASS_LABELS = CLASS_LABELS_dict[args.data]
#     dataset_name = args.data
# else:
#     if args.class_labels is None:
#         raise ValueError('If you want to train or test on a custom dataset,\
#                          you must also provide its class labels (in the correct\
#                          order.)')
#     else:
#         data = np.load(args.data_path, allow_pickle=True).item()
#         CLASS_LABELS = args.class_labels
#         dataset_name = Path(args.data_path).stem
#         args.data = dataset_name  # will be used for file naming
# x_source = data["x"]
# y_source = data["y"]


# if args.mode=="train":
#     model = TIMNET_Model(args=args, input_shape=x_source.shape[1:], class_label=CLASS_LABELS)
#     model.train(x_source, y_source)
# elif args.mode=='my_train':
#     model = TIMNET_Model(args=args, input_shape=x_source.shape[1:], class_label=CLASS_LABELS)
#     model.my_train(x_source, y_source)
# elif args.mode=="test":
#     model = TIMNET_Model(args=args, input_shape=x_source.shape[1:], class_label=CLASS_LABELS)
#     x_feats, y_labels = model.test(x_source, y_source, path=args.test_path)# x_feats and y_labels are test datas for t-sne
# ### MY MODIFICATION
# ### Run my_test defined in Model.py 
# elif args.mode=='my_test':
#     if args.weights_path is None:
#         raise ValueError('You must specify the path to the .hdf5 weights file\
#                          in order to test an isomorphic model on the provided\
#                          dataset.')
#     else:
#         my_test(args=args, x=x_source, y=y_source, class_labels=CLASS_LABELS,
#                 dataset_name=dataset_name)
  