import shap
import tensorflow as tf
# tf.compat.v1.enable_eager_execution()
# from keras.models import load_model
# from keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
tf.compat.v1.disable_v2_behavior()# solve shap's incompatibility 'TFDeep' object has no attribute 'between_tensors'.
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
import os
import matplotlib.pyplot as plt
import sys
import yaml

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
    splitter = model_selection.StratifiedShuffleSplit(n_splits=n_split, test_size=test_ratio, random_state=random_state)
    # get local indices, i.e. indices in [0, len(data_labels))
    train_indices, test_indices = zip(*splitter.split(X=np.zeros(len(id_list)), y=idwise_label_list))
    # return global datasets indices and labels
    train_ids = [id_list[fold_indices] for fold_indices in train_indices]
    test_ids = [id_list[fold_indices] for fold_indices in test_indices]
    train_labels = [idwise_label_list[fold_indices] for fold_indices in train_indices] # time_step wise labels
    test_labels = [idwise_label_list[fold_indices] for fold_indices in test_indices] # time_step wise labels
    return train_ids, train_labels, test_ids, test_labels

def prepare(dataset, model_path):
    '''prepare data and model for explaination'''
    ids = dataset.drop_duplicates(subset=['project'],keep='last')['project'].values
    X, y, id_sel = build_truncated_set(ids, time_step, feature_list, target_column, feature_scaler)
    model = load_model(model_path)
    return X, id_sel, model

def shapval_featuretimestep_boxplot(shap_values, id, id_sel, feature_list, path, showfliers=True, log_linthresh=None):
    sv = shap_values[1][id_sel.index(id)]# shape: (timestep, feature)
    # save shap values to csv file
    pd.DataFrame(sv, columns=feature_list).to_csv('{}/shapvalues_proj{}.csv'.format(path, id))
    # draw box graph using shap values
    plt.figure(figsize=(8,4),dpi=100)
    plt.boxplot(sv, showfliers=showfliers, whis=1.5, widths=0.5, patch_artist=True, boxprops=dict(facecolor='aquamarine', color='black'), medianprops=dict(color='black'), capprops=dict(color='black'), whiskerprops=dict(color='black'))
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    # plt.plot([1, len(tv_feature_names)+len(tinv_feature_names)], [0, 0], color='brown', linestyle='--', linewidth=1)
    plt.xticks(range(1, len(feature_list)+1), feature_list, rotation=90)
    plt.ylabel('Shap Value')
    if log_linthresh is not None:
        plt.yscale('symlog', linthresh=log_linthresh)
    plt.tight_layout()
    plt.savefig('{}/box_proj{}{}{}.png'.format(path, id, "_log" if log_linthresh is not None else "", "" if showfliers else "_removeouliers"), dpi=300, bbox_inches = 'tight')
    # plt.show()
def boxplot_main(dataset, sample_list, feature_list, model_path, save_path='./results'):
    for id in sample_list:
        X, id_sel, model = prepare(dataset, model_path)
        if id not in id_sel:
            print("proj{} seqlen shorter than 33".format(id))
            continue
        explainer = shap.DeepExplainer(model,X)
        shap_values = explainer.shap_values(X)
        print(np.array(shap_values).shape)# shape: (output, sample, timestep, feature)
        shapval_featuretimestep_boxplot(shap_values, id, id_sel, feature_list, save_path, showfliers=True, log_linthresh=1e-7)
        shapval_featuretimestep_boxplot(shap_values, id, id_sel, feature_list, save_path, showfliers=False, log_linthresh=1e-7)
        shapval_featuretimestep_boxplot(shap_values, id, id_sel, feature_list, save_path, showfliers=True)
    return shap_values# dict{cross_i:shap_values}, where shap_values is in shape (2, sample, timestep, feature)

def draw_linegraph(feature_sv, feature, path, show_all=False, abs=False):
    # draw features shap values changing trajectory through timestep
    if abs:
        feature_sv = np.abs(feature_sv)
    plt.plot(range(feature_sv.shape[0]+1), np.insert(feature_sv.mean(axis=1), 0, 0, axis=0) , color='#e0c0da', label='mean')
    if show_all:
        plt.fill_between(range(feature_sv.shape[0]+1), np.insert(feature_sv.min(axis=1), 0, 0, axis=0), np.insert(feature_sv.max(axis=1), 0, 0, axis=0), color='#c0e0c6', alpha=0.5, label='all')
    if abs:
        plt.ylabel('Abs Shap Value', fontsize=25)
    else:
        plt.ylabel('Shap Value', fontsize=25)
    plt.xlabel('Month', fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # plt.legend()
    plt.grid(linestyle='--')
    plt.tight_layout()
    plt.savefig('{}/line_feat{}{}{}.png'.format(path, feature, "_absval" if abs==True else "", "_showall" if show_all==True else ""), dpi=300, bbox_inches='tight')
    # plt.show()
def linegraph_main(dataset, feature_set, model_path, save_path="./results"):
    pred = 1 #0:dormant, 1:sustainable
    X, id_sel, model = prepare(dataset, model_path)
    explainer = shap.DeepExplainer(model,X)
    shap_values = explainer.shap_values(X)
    np.array(shap_values).shape# shape: (2, sample_num, timestep, feature_num)
    for feature in feature_set:
        sv = shap_values[pred].transpose(2,1,0)# shape:(feature_num, timestep, sample_num)
        feature_sv = sv[feature_list.index(feature)]# shape:(timestep, sample_num)
        draw_linegraph(feature_sv, feature, save_path, show_all=False, abs=True)
        draw_linegraph(feature_sv, feature, save_path, show_all=True, abs=True)
    return shap_values# shape: (2, sample_num, timestep, feature_num)
def stackbar_main(dataset, model_path, feature_list, save_path="./results"):
    pred = 1 #0:dormant, 1:sustainable

    X, id_sel, model = prepare(dataset, model_path)
    explainer = shap.DeepExplainer(model,X)
    shap_values = explainer.shap_values(X)

    sv = shap_values[pred].transpose(2,0,1)# shape:(feature_num, sample_num, timestep)
    sv = sv.sum(axis=-1)# shape:(feature_num, sample_num)
    stack_sv = []# shape:(feature_num, 2)
    for idx, sample_shaps in enumerate(sv):
        pos, neg, zero = 0, 0, 0
        for shapval in sample_shaps:
            if shapval>0:
                pos+=1
            elif shapval<0:
                neg+=1
            else:
                zero+=1
        print("feature#{}: pos#{} neg#{} zero#{}".format(idx, pos, neg, zero))
        stack_sv.append([pos, neg])

    plt.bar(range(len(feature_list)), np.array(stack_sv)[:,0], label='Positive', color='#FFBE7A')
    plt.bar(range(len(feature_list)), np.array(stack_sv)[:,1], bottom=np.array(stack_sv)[:,0], label='Negative', color='#82B0D2')
    plt.xticks(range(len(feature_list)), feature_list, rotation=90)
    plt.ylabel('#Projects', fontsize=25)
    plt.legend()
    plt.tight_layout()
    plt.savefig('{}/stackbar.png'.format(save_path), dpi=300, bbox_inches='tight')
    # plt.show()
    return stack_sv# shape:(feature_num, 2)

def prepare_features():
    project_features = ['active_devs','num_files','num_commits', 'c_percentage','inactive_c']
    technical_features =  ['c_nodes','c_edges','c_c_coef','c_mean_degree','c_long_tail']
    all_feature_names = []
    all_feature_names.extend(project_features)
    all_feature_names.extend(technical_features)
    return all_feature_names

if __name__ == "__main__":
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
    time_step = config['timestep']
    # feature set
    feature_list = prepare_features()
    target_column = 'label'
    feature_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(dataset[feature_list].values)
    mode = config['mode']# 'box', 'line', 'stackbar'

    model_path = config['model_path']
    output_path = config['output_path']
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    sample_list = config['sample_list']# project ids
    

    # explain&plot
    if mode == 'box':
        boxplot_main(dataset, sample_list, feature_list, model_path, output_path)
    elif mode == 'line':
        linegraph_main(dataset, feature_list, model_path, output_path)
    elif mode == 'stackbar':
        stackbar_main(dataset, model_path, feature_list, output_path)
    else:
        print("mode error")
        exit(0)
