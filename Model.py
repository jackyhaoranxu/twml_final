"""
@author: Jiaxin Ye
@contact: jiaxin-ye@foxmail.com
"""
import numpy as np
import tensorflow.keras.backend as K
import os
import tensorflow as tf
from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.keras import callbacks
from tensorflow.keras.layers import Layer,Dense,Input
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix
from Common_Model import Common_Model
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import datetime
import pandas as pd
from pathlib import Path

from TIMNET import TIMNET


def smooth_labels(labels, factor=0.1):
    # smooth the labels
    labels *= (1 - factor)
    labels += (factor / labels.shape[1])
    return labels

class WeightLayer(Layer):
    def __init__(self, **kwargs):
        super(WeightLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1],1),
                                      initializer='uniform',
                                      trainable=True)  
        super(WeightLayer, self).build(input_shape)  
 
    def call(self, x):
        tempx = tf.transpose(x,[0,2,1])
        x = K.dot(tempx,self.kernel)
        x = tf.squeeze(x,axis=-1)
        return  x
 
    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[2])
    
def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex/K.sum(ex, axis=axis, keepdims=True)

class TIMNET_Model(Common_Model):
    def __init__(self, args, input_shape, class_label, **params):
        super(TIMNET_Model,self).__init__(**params)
        self.args = args
        self.data_shape = input_shape
        self.num_classes = len(class_label)
        self.class_label = class_label
        self.matrix = []
        self.eva_matrix = []
        self.acc = 0
        print("TIMNET MODEL SHAPE:",input_shape)
    
    def create_model(self):
        self.inputs=Input(shape = (self.data_shape[0],self.data_shape[1]))
        self.multi_decision = TIMNET(nb_filters=self.args.filter_size,
                                kernel_size=self.args.kernel_size, 
                                nb_stacks=self.args.stack_size,
                                dilations=self.args.dilation_size,
                                dropout_rate=self.args.dropout,
                                activation = self.args.activation,
                                return_sequences=True, 
                                name='TIMNET')(self.inputs)

        self.decision = WeightLayer()(self.multi_decision)
        self.predictions = Dense(self.num_classes, activation='softmax')(self.decision)
        self.model = Model(inputs = self.inputs, outputs = self.predictions)
        
        self.model.compile(loss = "categorical_crossentropy",
                           optimizer =Adam(learning_rate=self.args.lr, beta_1=self.args.beta1, beta_2=self.args.beta2, epsilon=1e-8),
                           metrics = ['accuracy'])
        print("Temporal create succes!")
        
    def train(self, x, y):

        filepath = self.args.model_path
        resultpath = self.args.result_path

        if not os.path.exists(filepath):
            os.mkdir(filepath)
        if not os.path.exists(resultpath):
            os.mkdir(resultpath)

        i=1
        now = datetime.datetime.now()
        now_time = datetime.datetime.strftime(now,'%Y-%m-%d_%H-%M-%S')
        kfold = KFold(n_splits=self.args.split_fold, shuffle=True, random_state=self.args.random_seed)
        avg_accuracy = 0
        avg_loss = 0
        for train, test in kfold.split(x, y):
            self.create_model()
            y[train] = smooth_labels(y[train], 0.1)
            ### TODO: Match output folder name to dataset name
            folder_address = filepath+self.args.data+"_"+str(self.args.random_seed)+"_"+now_time
            if not os.path.exists(folder_address):
                os.mkdir(folder_address)
            weight_path=folder_address+'/'+str(self.args.split_fold)+"-fold_weights_best_"+str(i)+".hdf5"
            checkpoint = callbacks.ModelCheckpoint(weight_path, monitor='val_accuracy', verbose=1,save_weights_only=True,save_best_only=True,mode='max')
            max_acc = 0
            best_eva_list = []
            h = self.model.fit(x[train], y[train],validation_data=(x[test],  y[test]),batch_size = self.args.batch_size, epochs = self.args.epoch, verbose=1,callbacks=[checkpoint])
            self.model.load_weights(weight_path)
            best_eva_list = self.model.evaluate(x[test],  y[test])
            avg_loss += best_eva_list[0]
            avg_accuracy += best_eva_list[1]
            print(str(i)+'_Model evaluation: ', best_eva_list,"   Now ACC:",str(round(avg_accuracy*10000)/100/i))
            i+=1
            y_pred_best = self.model.predict(x[test])
            self.matrix.append(confusion_matrix(np.argmax(y[test],axis=1),np.argmax(y_pred_best,axis=1)))
            em = classification_report(np.argmax(y[test],axis=1),np.argmax(y_pred_best,axis=1), target_names=self.class_label,output_dict=True)
            self.eva_matrix.append(em)
            print(classification_report(np.argmax(y[test],axis=1),np.argmax(y_pred_best,axis=1), target_names=self.class_label))

        print("Average ACC:",avg_accuracy/self.args.split_fold)
        self.acc = avg_accuracy/self.args.split_fold
        writer = pd.ExcelWriter(resultpath+self.args.data+'_'+str(self.args.split_fold)+'fold_'+str(round(self.acc*10000)/100)+"_"+str(self.args.random_seed)+"_"+now_time+'.xlsx')
        for i,item in enumerate(self.matrix):
            temp = {}
            temp[" "] = self.class_label
            for j,l in enumerate(item):
                temp[self.class_label[j]]=item[j]
            data1 = pd.DataFrame(temp)
            data1.to_excel(writer,sheet_name=str(i), encoding='utf8')

            df = pd.DataFrame(self.eva_matrix[i]).transpose()
            df.to_excel(writer,sheet_name=str(i)+"_evaluate", encoding='utf8')
        writer.save()
        writer.close()

        K.clear_session()
        self.matrix = []
        self.eva_matrix = []
        self.acc = 0
        self.trained = True

    ### MY TRAIN, using train_test_split, generated by chatgpt
    def my_train(self, x, y):

        filepath = self.args.model_path
        resultpath = self.args.result_path

        if not os.path.exists(filepath):
            os.mkdir(filepath)
        if not os.path.exists(resultpath):
            os.mkdir(resultpath)

        self.create_model()
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=self.args.random_seed, shuffle=True, stratify=y)
        y_train = smooth_labels(y_train, 0.1)

        now = datetime.datetime.now()
        now_time = datetime.datetime.strftime(now,'%Y-%m-%d_%H-%M-%S')

        folder_address = filepath+self.args.data+"_"+str(self.args.random_seed)+"_"+now_time
        if not os.path.exists(folder_address):
            os.mkdir(folder_address)

        weight_path = folder_address+'/'+"weights_best.hdf5"

        checkpoint = callbacks.ModelCheckpoint(weight_path, monitor='val_accuracy', verbose=1,save_weights_only=True,save_best_only=True,mode='max')

        self.model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size = self.args.batch_size, epochs = self.args.epoch, verbose=1, callbacks=[checkpoint])

        self.model.load_weights(weight_path)
        best_eva_list = self.model.evaluate(x_test, y_test)

        print('Model evaluation: ', best_eva_list)

        y_pred_best = self.model.predict(x_test)
        self.matrix.append(confusion_matrix(np.argmax(y_test,axis=1),np.argmax(y_pred_best,axis=1)))

        em = classification_report(np.argmax(y_test,axis=1),np.argmax(y_pred_best,axis=1), target_names=self.class_label,output_dict=True)
        self.eva_matrix.append(em)

        print(classification_report(np.argmax(y_test,axis=1),np.argmax(y_pred_best,axis=1), target_names=self.class_label))

        writer = pd.ExcelWriter(resultpath+self.args.data+'_'+str(round(best_eva_list[1]*10000)/100)+"_"+self.args.data_type+str(self.args.random_seed)+"_"+now_time+'.xlsx')

        temp = {}
        temp[" "] = self.class_label
        for j,l in enumerate(self.matrix[0]):
            temp[self.class_label[j]]=self.matrix[0][j]
        data1 = pd.DataFrame(temp)
        data1.to_excel(writer,sheet_name="0", encoding='utf8')

        df = pd.DataFrame(self.eva_matrix[0]).transpose()
        df.to_excel(writer,sheet_name="0_evaluate", encoding='utf8')
        
        writer.save()
        writer.close()

        K.clear_session()
        self.matrix = []
        self.eva_matrix = []
        self.trained = True
        
        return self.model
    
    ######## THIS VERSION IS ONLY COMPATIBLE WITH A SINGLE DATASET
    # def my_train(self, x, y):

    #     filepath = self.args.model_path
    #     resultpath = self.args.result_path

    #     if not os.path.exists(filepath):
    #         os.mkdir(filepath)
    #     if not os.path.exists(resultpath):
    #         os.mkdir(resultpath)

    #     self.create_model()
    #     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=self.args.random_seed)
    #     y_train = smooth_labels(y_train, 0.1)

    #     now = datetime.datetime.now()
    #     now_time = datetime.datetime.strftime(now,'%Y-%m-%d_%H-%M-%S')

    #     folder_address = filepath+self.args.data+"_"+str(self.args.random_seed)+"_"+now_time
    #     if not os.path.exists(folder_address):
    #         os.mkdir(folder_address)

    #     weight_path = folder_address+'/'+"weights_best.hdf5"

    #     checkpoint = callbacks.ModelCheckpoint(weight_path, monitor='val_accuracy', verbose=1,save_weights_only=True,save_best_only=True,mode='max')

    #     self.model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size = self.args.batch_size, epochs = self.args.epoch, verbose=1, callbacks=[checkpoint])

    #     self.model.load_weights(weight_path)
    #     best_eva_list = self.model.evaluate(x_test, y_test)

    #     print('Model evaluation: ', best_eva_list)

    #     y_pred_best = self.model.predict(x_test)
    #     self.matrix.append(confusion_matrix(np.argmax(y_test,axis=1),np.argmax(y_pred_best,axis=1)))

    #     em = classification_report(np.argmax(y_test,axis=1),np.argmax(y_pred_best,axis=1), target_names=self.class_label,output_dict=True)
    #     self.eva_matrix.append(em)

    #     print(classification_report(np.argmax(y_test,axis=1),np.argmax(y_pred_best,axis=1), target_names=self.class_label))

    #     writer = pd.ExcelWriter(resultpath+self.args.data+'_'+str(round(best_eva_list[1]*10000)/100)+"_"+str(self.args.random_seed)+"_"+now_time+'.xlsx')

    #     temp = {}
    #     temp[" "] = self.class_label
    #     for j,l in enumerate(self.matrix[0]):
    #         temp[self.class_label[j]]=self.matrix[0][j]
    #     data1 = pd.DataFrame(temp)
    #     data1.to_excel(writer,sheet_name="0", encoding='utf8')

    #     df = pd.DataFrame(self.eva_matrix[0]).transpose()
    #     df.to_excel(writer,sheet_name="0_evaluate", encoding='utf8')
        
    #     writer.save()
    #     writer.close()

    #     K.clear_session()
    #     self.matrix = []
    #     self.eva_matrix = []
    #     self.trained = True


    
    def test(self, x, y, path):
        i=1
        kfold = KFold(n_splits=self.args.split_fold, shuffle=True, random_state=self.args.random_seed)
        avg_accuracy = 0
        avg_loss = 0
        x_feats = []
        y_labels = []
        for train, test in kfold.split(x, y):
            self.create_model()
            weight_path=path+'/'+str(self.args.split_fold)+"-fold_weights_best_"+str(i)+".hdf5"
            self.model.fit(x[train], y[train],validation_data=(x[test],  y[test]),batch_size = 64,epochs = 0,verbose=0)
            self.model.load_weights(weight_path)#+source_name+'_single_best.hdf5')
            best_eva_list = self.model.evaluate(x[test],  y[test])
            avg_loss += best_eva_list[0]
            avg_accuracy += best_eva_list[1]
            print(str(i)+'_Model evaluation: ', best_eva_list,"   Now ACC:",str(round(avg_accuracy*10000)/100/i))
            i+=1
            y_pred_best = self.model.predict(x[test])
            self.matrix.append(confusion_matrix(np.argmax(y[test],axis=1),np.argmax(y_pred_best,axis=1)))
            em = classification_report(np.argmax(y[test],axis=1),np.argmax(y_pred_best,axis=1), target_names=self.class_label,output_dict=True)
            self.eva_matrix.append(em)
            print(classification_report(np.argmax(y[test],axis=1),np.argmax(y_pred_best,axis=1), target_names=self.class_label))
            caps_layer_model = Model(inputs=self.model.input,
            outputs=self.model.get_layer(index=-2).output)
            feature_source = caps_layer_model.predict(x[test])
            x_feats.append(feature_source)
            y_labels.append(y[test])
        print("Average ACC:",avg_accuracy/self.args.split_fold)
        self.acc = avg_accuracy/self.args.split_fold
        return x_feats, y_labels
    

### ------------------------------------------------------------------------
### MY MODIFICATION:
### Tests the model with input weights on the loaded dataset,
### and save the confusion matrix and classification report 
### as .npy and .csv respectively.
def create_timnet_model(args, data_shape, num_classes):
    inputs = Input(shape = (data_shape[0], data_shape[1]))
    multi_decision = TIMNET(nb_filters=args.filter_size,
                            kernel_size=args.kernel_size, 
                            nb_stacks=args.stack_size,
                            dilations=args.dilation_size,
                            dropout_rate=args.dropout,
                            activation = args.activation,
                            return_sequences=True, 
                            name='TIMNET')(inputs)

    decision = WeightLayer()(multi_decision)
    predictions = Dense(num_classes, activation='softmax')(decision)
    model = Model(inputs = inputs, outputs = predictions)

    model.compile(loss = "categorical_crossentropy",
                   optimizer =Adam(learning_rate=args.lr, beta_1=args.beta1, 
                                   beta_2=args.beta2, epsilon=1e-8),
                   metrics = ['accuracy'])
    print("Temporal create success!")
    return model


def my_test(args, model, x, y, class_labels, dataset_name): 
    ### CAVEAT: Must ensure that the data on which the weights are trained 
    ### has the same x and y shapes as the test data.
    # weights_path = args.weights_path
    results_folder = args.result_path

    # # Create the model and load weights
    # num_classes = len(class_labels)
    # data_shape = x.shape[1:]
    # model = create_timnet_model(args=args, data_shape=data_shape, 
    #                             num_classes=num_classes)
    # model.load_weights(weights_path)

    # Get the predictions and evaluate
    y_pred = model.predict(x)
    y_true = np.argmax(y, axis=1)
    y_pred_best = np.argmax(y_pred, axis=1)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred_best)
    conf_matrix_df = pd.DataFrame(conf_matrix, columns=class_labels,
                                  index=class_labels)
    print('Confusion Matrix: \n', conf_matrix_df)

    # Save confusion matrix as .csv
    # model_name = Path(weights_path).stem
    # test_name = model_name + "_ON_" + dataset_name
    save_folder = Path(results_folder)
    conf_matrix_path = save_folder / ("CONF_MATRIX_" + args.data + '_' + args.data_type + ".csv")
    conf_matrix_df.to_csv(conf_matrix_path)
    print(f'Confusion matrix has been saved to {conf_matrix_path}')

    # Classification report
    eval_metrics = classification_report(y_true, y_pred_best, 
                                         target_names=class_labels, 
                                         output_dict=True)
    eval_metrics_df = pd.DataFrame(eval_metrics).transpose()
    print('Classification Report: \n', eval_metrics_df)

    # Save classification report as .csv
    eval_metrics_path = save_folder / ("EVAL_METRICS_" + args.data + '_' + args.data_type + ".csv")
    eval_metrics_df.to_csv(eval_metrics_path)
    print(f'Classification report has been saved to {eval_metrics_path}')

    return


######## THIS VERSION IS ONLY COMPATIBLE WITH A SINGLE DATASET
# def my_test(args, x, y, class_labels, dataset_name): 
#     ### CAVEAT: Must ensure that the data on which the weights are trained 
#     ### has the same x and y shapes as the test data.
#     weights_path = args.weights_path
#     results_folder = args.result_path

#     # Create the model and load weights
#     num_classes = len(class_labels)
#     data_shape = x.shape[1:]
#     model = create_timnet_model(args=args, data_shape=data_shape, 
#                                 num_classes=num_classes)
#     model.load_weights(weights_path)

#     # Get the predictions and evaluate
#     y_pred = model.predict(x)
#     y_true = np.argmax(y, axis=1)
#     y_pred_best = np.argmax(y_pred, axis=1)
    
#     # Confusion matrix
#     conf_matrix = confusion_matrix(y_true, y_pred_best)
#     conf_matrix_df = pd.DataFrame(conf_matrix, columns=class_labels,
#                                   index=class_labels)
#     print('Confusion Matrix: \n', conf_matrix_df)

#     # Save confusion matrix as .csv
#     model_name = Path(weights_path).stem
#     test_name = model_name + "_ON_" + dataset_name
#     save_folder = Path(results_folder)
#     conf_matrix_path = save_folder / ("CONF_MATRIX_" + test_name + ".csv")
#     conf_matrix_df.to_csv(conf_matrix_path)
#     print(f'Confusion matrix has been saved to {conf_matrix_path}')

#     # Classification report
#     eval_metrics = classification_report(y_true, y_pred_best, 
#                                          target_names=class_labels, 
#                                          output_dict=True)
#     eval_metrics_df = pd.DataFrame(eval_metrics).transpose()
#     print('Classification Report: \n', eval_metrics_df)

#     # Save classification report as .csv
#     eval_metrics_path = save_folder / ("EVAL_METRICS_" + test_name + ".csv")
#     eval_metrics_df.to_csv(eval_metrics_path)
#     print(f'Classification report has been saved to {eval_metrics_path}')

#     return