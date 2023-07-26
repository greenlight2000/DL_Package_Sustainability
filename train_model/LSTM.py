# -*- coding: utf-8 -*-
# Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf

import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn import model_selection
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score, precision_recall_curve, auc, roc_curve
# from sklearn.model_selection import train_test_split
import datetime
import os
import yaml
import sys

print(tf.__version__)# 2.12.0
print(np.__version__)# 1.23.5
print(pd.__version__)# 1.5.3
print(sklearn.__version__)# 1.2.2

def build_truncated_set(ids, seq_len, feature_names, target_column, feature_scaler):
    """
    truncate variable length data to fixed length `seq_len` and scale tv features
    """
    X = []# [proj_num,seq_len,feat_dim]
    y = []# [project_num, cat_num] project-wise label, not timestep-wise label
    selected_ids = []# [project_num]
    for id in ids:
        seq = dataset.query('project=={}'.format(id))
        if seq.shape[0] < seq_len:
            continue
        seq.reset_index(inplace=True)
        X.append(feature_scaler.transform(seq.loc[:seq_len-1][feature_names]))
        y.append([seq.loc[0][target_column]])
        selected_ids.append(id)
    return np.array(X), to_categorical(np.array(y,dtype=int)), selected_ids

def IDsStratifiedShuffleSplit(id_list, idwise_label_list, test_ratio=None, n_split=1, random_state=1337):
    '''stratified data splitter'''
    splitter = model_selection.StratifiedShuffleSplit(n_splits=n_split, test_size=test_ratio, random_state=random_state)
    # get local indices, i.e. indices in [0, len(data_labels))
    train_indices, test_indices = zip(*splitter.split(X=np.zeros(len(id_list)), y=idwise_label_list))
    # return global datasets indices and labels
    train_ids = [id_list[fold_indices] for fold_indices in train_indices]
    test_ids = [id_list[fold_indices] for fold_indices in test_indices]
    train_labels = [idwise_label_list[fold_indices] for fold_indices in train_indices] # time_step wise labels
    test_labels = [idwise_label_list[fold_indices] for fold_indices in test_indices] # time_step wise labels
    return train_ids, train_labels, test_ids, test_labels

# calculate AUPRC metrics for keras model
def AUPRC(y_true, y_pred):
    # calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    # calculate precision-recall AUC
    return auc(recall, precision)

# calculate AUROC metrics for keras model
def AUROC(y_true, y_pred):
    # calculate precision-recall curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    # calculate precision-recall AUC
    return auc(fpr, tpr)

class Metrics(keras.callbacks.Callback):
    def __init__(self, valid_data):
        super(Metrics, self).__init__()
        self.validation_data = valid_data
        self.val_f1_scores = []
 
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_predict = np.argmax(self.model.predict(self.validation_data[0]), axis=1)
        val_targ = np.argmax(self.validation_data[1], axis=1)
        if len(val_targ.shape) == 2 and val_targ.shape[1] != 1:
            val_targ = np.argmax(val_targ, -1)
 
        _val_f1 = f1_score(val_targ, val_predict, average='weighted')
        _val_recall = recall_score(val_targ, val_predict, average='weighted')
        _val_precision = precision_score(val_targ, val_predict, average='weighted')
        self.val_f1_scores.append(_val_f1)
 
        logs['val_f1'] = _val_f1
        logs['val_recall'] = _val_recall
        logs['val_precision'] = _val_precision
        # print(" — val_f1: %f — val_precision: %f — val_recall: %f" % (_val_f1, _val_precision, _val_recall))
        return

class LSTM_model():
    def __init__(self, exp_name, input_shape, LSTM_dim=64, LSTM_layers=1, dropout=0.3, loss='binary_crossentropy', optimizer=Adam()):
        self.time_step, self.feat_dim = input_shape
        self.dropout = dropout
        self.LSTM_layers = LSTM_layers
        self.LSTM_dim = LSTM_dim
        self.train_history = None
        self.eval_metrics = None
        self.best_model_path = None
        self.name=exp_name

        input = Input(shape=(self.time_step, self.feat_dim))
        output = LSTM(self.LSTM_dim)(input)# shape: (batch_size, output_dim)
        output = Dropout(self.dropout)(output)
        for i in range(self.LSTM_layers-1):
            output = LSTM(self.LSTM_dim, return_sequences=True)(output)
            output = Dropout(self.dropout)(output)
        classi_output = Dense(2, activation='softmax')(output)

        model = Model(inputs=input, outputs=classi_output)
        model.compile(loss=loss, optimizer=optimizer, metrics=['binary_accuracy'])
        self.model = model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, output_path='./exp1'):
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        package_path = '{}/train_records/{}/'.format(output_path, self.time_step)
        if not os.path.exists(package_path):
            os.makedirs(package_path)

        model_name="bestmodel_{}_{}@{}.hdf5".format(str(self.time_step), self.name, nowtime)
        fig_name = 'fig_{}_{}@{}.png'.format(self.time_step, self.name, nowtime)
        metrics = Metrics(valid_data=(X_val, y_val))
        checkpoint = ModelCheckpoint(package_path+model_name, monitor='val_binary_accuracy', mode='max', verbose=0, save_best_only=True)
        history = self.model.fit(X_train, y_train, batch_size=10, epochs=epochs, verbose=0, validation_data=(X_val,y_val), callbacks=[metrics, checkpoint])
        self.train_history = history
        self.best_model_path = package_path+model_name
        plt.figure(figsize=(8,4),dpi=100)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.plot(history.history['binary_accuracy'],':')
        plt.plot(history.history['val_binary_accuracy'],':')
        plt.plot(metrics.val_f1_scores,'--')
        plt.title('model train vs validation loss/metrics')
        plt.ylabel('metrics')
        plt.xlabel('epoch')
        plt.legend(['loss', 'val_loss', 'train acc','validation acc', 'val_f1'], loc='upper right')
        plt.savefig(package_path+fig_name)
        # plt.show()
        return history
    
    def evaluate(self, X_test, y_test, model_path=None, mode='best', output_path='./exp1'):
        if model_path is not None:
            model = load_model(model_path)
        elif mode=='best':
            model = load_model(self.best_model_path)
        elif mode=='last':
            model = self.model
        else:
            raise Exception('unknown mode. please input "best" or "last"')
        y_pred = np.argmax(model.predict(X_test), axis=1)
        y_true = np.argmax(y_test, axis=1)
        metrics = classification_report(y_true, y_pred, output_dict = True)
        _AUROC = AUROC(y_true, y_pred)
        _AUPRC = AUPRC(y_true, y_pred)
        self.eval_metrics = metrics
        output = '{}/metrics_{}.csv'.format(output_path, self.name)
        if not os.path.exists(output):
            with open(output, 'a') as f:
                f.write('time_step,acc,precision,recall,f1,sustainable_f1,dormant_f1,auroc,auprc,model_path\n')
        with open(output, 'a') as f:
            t_month = str(self.time_step)
            acc = str(metrics['accuracy'])
            precision =  str(metrics['weighted avg']['precision'])
            recall = str(metrics['weighted avg']['recall'])
            f1 =  str(metrics['weighted avg']['f1-score'])
            sustainable_f1 = str(metrics['1']['f1-score']) if '1' in metrics else '0'
            dormant_f1 =  str(metrics['0']['f1-score']) if '0' in metrics else '0'
            auroc = str(_AUROC)
            auprc = str(_AUPRC)
            path = self.best_model_path if mode=='best' else ''
            things = [t_month,acc,precision,recall,f1,sustainable_f1,dormant_f1,auroc,auprc,path]
            f.write(','.join(things))
            f.write('\n')
        return metrics

def prepare_features():
    project_features = ['active_devs','num_files','num_commits', 'c_percentage','inactive_c']
    technical_features =  ['c_nodes','c_edges','c_c_coef','c_mean_degree','c_long_tail']
    all_feature_names = []
    all_feature_names.extend(project_features)
    all_feature_names.extend(technical_features)
    return all_feature_names

def get_optimizer(text):
    if text=='Adam':
        return Adam()
    else:
        raise Exception('unknown optimizer, please input "Adam"')
    
def prepare_LSTM_timestep(dataset, start, end):
    start_timestep, end_timestep = 0, -1
    # start
    if isinstance(start, int):
        start_timestep = start
    elif isinstance(start, str):
        proj_months = dataset.drop_duplicates(subset=['project'],keep='last')['month']
        if start == "min":
            start_timestep = int(proj_months.min())# all projects can be used for training
        elif start == "quantile1":
            start_timestep = int(proj_months.quantile(0.25))# 3/4 of projects can be used for training
        elif start == "median":
            start_timestep = int(proj_months.median())# 1/2 of projects can be used for training
        elif start == "quantile3":
            start_timestep = int(proj_months.quantile(0.75))# 1/4 of projects can be used for training
        else:
            raise Exception("unknown start str, please input one of 'min', 'quantile1', 'median', 'quantile3'")
    else:
        raise Exception("unknown start type, please input one of int, str")
    # end
    if isinstance(end, int):
        end_timestep = end
    elif isinstance(end, str):
        proj_months = dataset.drop_duplicates(subset=['project'],keep='last')['month']
        if end == "min":
            end_timestep = int(proj_months.min())# all projects can be used for training
        elif end == "quantile1":
            end_timestep = int(proj_months.quantile(0.25))# 3/4 of projects can be used for training
        elif end == "median":
            end_timestep = int(proj_months.median())# 1/2 of projects can be used for training
        elif end == "quantile3":
            end_timestep = int(proj_months.quantile(0.75))
        # elif end == "max":
        #     end_timestep = int(proj_months.max())
        else:
            raise Exception("unknown end str, please input one of 'min', 'quantile1', 'median', 'quantile3'")
    else:
        raise Exception("unknown end type, please input one of int, str")
    if end_timestep < start_timestep:
        raise Exception("end_timestep should be larger than start_timestep")
    return start_timestep, end_timestep


if __name__ == '__main__':
    # read configurations from yml
    config_path = sys.argv[1]# './config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print(config)
    # seed
    seed=config['seed']
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # data
    dataset_path = config['dataset_path']
    dataset = pd.read_csv(dataset_path)
    dataset.replace('sustainable', '1', inplace=True) 
    dataset.replace('dormant', '0', inplace=True) 
    train_test_split_ratio = config['trainval_test_split_ratio']
    train_val_split_ratio = config['train_val_split_ratio']
    target_column = 'label'
    all_feature_names = prepare_features()
    feature_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(dataset[all_feature_names].values)
    start_timestep, end_timestep = prepare_LSTM_timestep(dataset, config['start_timestep'], config['end_timestep'])
    # model
    loss = config['loss']
    optimizer = config['optimizer']
    LSTM_hidden_size = config['LSTM_hidden_size']
    LSTM_num_layers = config['LSTM_num_layers']
    dropout = config['dropout']
    feature_dim = len(all_feature_names)
    # exp
    output_path = config['output_path']# all output will be palced under this folder
    exp_name = config['exp_name']# some summative files will contain this string
    epochs = config['epochs']# 50
    repeat_times = config['repeat_times']# exp repeat times，5

    # split dataset
    project_label_li = dataset.drop_duplicates(subset=['project'],keep='last')[['project','label']]
    ids, labels = project_label_li['project'].values, project_label_li['label'].values
    train_val_ids, train_val_labels, test_ids, test_labels = IDsStratifiedShuffleSplit(ids, labels, train_test_split_ratio, n_split=1)
    train_ids, train_labels, val_ids, val_labels = IDsStratifiedShuffleSplit(train_val_ids[0], train_val_labels[0], train_val_split_ratio, n_split=1)
    train_ids, val_ids, test_ids = train_ids[0], val_ids[0], test_ids[0]
    train_labels, val_labels, test_labels = train_labels[0], val_labels[0], test_labels[0]

    # training and evaluation
    models = []
    for time_step in range(start_timestep, end_timestep+1):
        X_train, y_train,_ = build_truncated_set(train_ids, time_step, all_feature_names, target_column, feature_scaler)
        X_val, y_val,_ = build_truncated_set(val_ids, time_step, all_feature_names, target_column, feature_scaler)
        X_test, y_test,_ = build_truncated_set(test_ids, time_step, all_feature_names, target_column, feature_scaler)
        
        for _ in range(repeat_times):
            model = LSTM_model(exp_name=exp_name, input_shape=(time_step, feature_dim), LSTM_dim=LSTM_hidden_size, LSTM_layers=LSTM_num_layers, dropout=dropout, loss=loss, optimizer=get_optimizer(optimizer))
            model.train(X_train, y_train, X_val, y_val, epochs=epochs, output_path=output_path)
            model.evaluate(X_test, y_test, mode='best', output_path=output_path)
        models.append(model)
            
    # result analysis
    if not os.path.exists(output_path+'/train_records/'):
        os.makedirs(output_path+'/train_records/')
    with open('{}/log_{}.txt'.format(config['output_path'],config['exp_name']), 'a') as f:
        f.write('config:\n{}\n'.format(config))
        f.write('feature_names:\n{}\n'.format(all_feature_names))
        f.write('model_arch@first ts:\n')
        models[0].model.summary(print_fn=lambda x: f.write(x + '\n'))
        f.write('model_arch@last ts:\n')
        models[-1].model.summary(print_fn=lambda x: f.write(x + '\n'))
    start_month, end_month = start_timestep, end_timestep
    month_metrics = pd.read_csv('{}/metrics_{}.csv'.format(config['output_path'],config['exp_name']))
    month_metrics.set_index('time_step', inplace=True)
    all = month_metrics.loc[list(range(start_month, end_month+1))]
    acc = "%.4f+-%.4f" % (all['acc'].mean(),all['acc'].max()-all['acc'].mean())
    prec = "%.4f+-%.4f" % (all['precision'].mean(),all['precision'].max()-all['precision'].mean())
    rec = "%.4f+-%.4f" % (all['recall'].mean(),all['recall'].max()-all['recall'].mean())
    f1 = "%.4f+-%.4f" % (all['f1'].mean(),all['f1'].max()-all['f1'].mean())
    auroc = "%.4f+-%.4f" % (all['auroc'].mean(),all['auroc'].max()-all['auroc'].mean())
    auprc = "%.4f+-%.4f" % (all['auprc'].mean(),all['auprc'].max()-all['auprc'].mean())
    with open('{}/log_{}.txt'.format(config['output_path'],config['exp_name']), 'a') as f:
        f.write('{}-{} month\'s performance average: \nacc: {}\nprec: {}\nrec: {}\nf1: {}\nauroc: {}\nauprc: {}\n--------------------------\n'.format(start_month, end_month, acc,prec,rec,f1,auroc,auprc))
    print('{}-{} month\'s performance average: \nacc: {}\nprec: {}\nrec: {}\nf1: {}\nauroc: {}\nauprc: {}\n--------------------------\n'.format(start_month, end_month, acc,prec,rec,f1,auroc,auprc))
    best_f1_ts = 0
    best_f1 = 0
    month_track = {'month':[],'acc_min':[],'acc_max':[],'acc_mean':[],'f1_min':[],'f1_max':[],'f1_mean':[],'prec_min':[],'prec_max':[],'prec_mean':[]}
    for i in range(start_month,end_month+1):
        metrics = month_metrics.loc[i]
        if metrics['f1'].mean()>best_f1:
            best_f1 = metrics['f1'].mean()
            best_f1_ts = i
        f1 = "%.4f+-%.4f" % (metrics['f1'].mean(),metrics['f1'].max()-metrics['f1'].mean())
        print('time_step: {}, f1: {}'.format(i, f1))
        month_track['month'].append(i)
        month_track['acc_min'].append(metrics['acc'].min())
        month_track['acc_max'].append(metrics['acc'].max())
        month_track['acc_mean'].append(metrics['acc'].mean())
        month_track['f1_min'].append(metrics['f1'].min())
        month_track['f1_max'].append(metrics['f1'].max())
        month_track['f1_mean'].append(metrics['f1'].mean())
        month_track['prec_min'].append(metrics['precision'].min())
        month_track['prec_max'].append(metrics['precision'].max())
        month_track['prec_mean'].append(metrics['precision'].mean())
    metrics = month_metrics.loc[best_f1_ts]
    acc = "%.4f+-%.4f" % (metrics['acc'].mean(),metrics['acc'].max()-metrics['acc'].mean())
    prec = "%.4f+-%.4f" % (metrics['precision'].mean(),metrics['precision'].max()-metrics['precision'].mean())
    rec = "%.4f+-%.4f" % (metrics['recall'].mean(),metrics['recall'].max()-metrics['recall'].mean())
    f1 = "%.4f+-%.4f" % (metrics['f1'].mean(),metrics['f1'].max()-metrics['f1'].mean())
    auroc = "%.4f+-%.4f" % (metrics['auroc'].mean(),metrics['auroc'].max()-metrics['auroc'].mean())
    auprc = "%.4f+-%.4f" % (metrics['auprc'].mean(),metrics['auprc'].max()-metrics['auprc'].mean())
    with open('{}/log_{}.txt'.format(config['output_path'],config['exp_name']), 'a') as f:
        f.write('best f1 at month: {}\nacc: {}\nprec: {}\nrec: {}\nf1: {}\nauroc: {}\nauprc: {}\n'.format(best_f1_ts,acc,prec,rec,f1,auroc,auprc))
    print('best f1 at month: {}\nacc: {}\nprec: {}\nrec: {}\nf1: {}\nauroc: {}\nauprc: {}\n'.format(best_f1_ts,acc,prec,rec,f1,auroc,auprc))
    # draw metric track in line graph
    plt.figure(figsize=(8,4),dpi=100)
    # plt.title('model performance track')
    plt.plot(month_track['month'],month_track['f1_mean'])
    plt.fill_between(month_track['month'], month_track['f1_min'], month_track['f1_max'], alpha=0.2) 
    plt.plot(month_track['month'],month_track['acc_mean'])
    plt.fill_between(month_track['month'], month_track['acc_min'], month_track['acc_max'], alpha=0.2)
    plt.plot(month_track['month'],month_track['prec_mean'])
    plt.fill_between(month_track['month'], month_track['prec_min'], month_track['prec_max'], alpha=0.2) 
    plt.legend(['F1','Accuracy','Precision'], loc='upper left')
    plt.xlabel('Month', fontsize=25)
    plt.yticks(fontsize=25)
    plt.xticks(fontsize=25)
    plt.tight_layout()
    plt.savefig('{}/metrictrack_{}.png'.format(config['output_path'],config['exp_name']))
    # plt.show()