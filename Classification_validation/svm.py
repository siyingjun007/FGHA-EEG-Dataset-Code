import argparse
import random
import time
from sklearn.svm import LinearSVC
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, precision_score, recall_score, confusion_matrix
import os
import joblib
import h5py
import warnings
from collections import Counter
def get_all_files(directory):
    all_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files
def load_data(data_paths):
    data = []
    labels = []
    for idx, file in enumerate(data_paths):
        s = get_all_files(os.path.join(data_path, file))
        datafiles = sorted(s)
        data_sub = None
        label_sub = None
        for datafile in datafiles:
            if 'label' in datafile:
                with h5py.File(datafile, 'r') as f:
                    labelraw = f['label'][:]
                    print(f"Label raw shape:", labelraw.shape)
                if label_sub is None:
                    label_sub = labelraw
                else:
                    label_sub = np.concatenate((label_sub, labelraw), axis=0)
            elif 'latency' not in datafile and 'conti' not in datafile:
                # print(datafile)
                with h5py.File(datafile, 'r') as f:
                    raw = f['EEGtemp']['data'][:]
                    raw = raw.T
                    # print(f"Raw shape:", raw.shape)
                    if data_sub is None:
                        data_sub = raw
                    else:
                        data_sub = np.concatenate((data_sub, raw), axis=2)
        if idx == 0:
            labels = label_sub
            data = data_sub
        else:
            labels = np.concatenate((labels, label_sub), axis=0)
            data = np.concatenate((data, data_sub), axis=2)

    return data, labels
def preprocess_test_data(test_data, test_labels, label_map):
    test_data = (test_data.reshape(test_data.shape[0], test_data.shape[1] * test_data.shape[2]))
    print(test_data.shape)
    test_labels  = np.vectorize(label_map.get)(test_labels)
    return test_data, test_labels
def train_svm(best_fold,data, labels):
    # print(f" data shape:", data.shape)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    fold = -1
    for train_index, test_index in skf.split(data, labels):
        fold = fold+1
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test =labels[train_index], labels[test_index]
        print('fold', fold)
        data_train= X_train
        data_val =X_test
        label_train=y_train
        label_val = y_test
        # print('data_train: ', data_train.shape)
        # print('data_val: ', data_val.shape)
        # print('label_train: ', label_train.shape)
        # print('label_val: ', label_val.shape)
        label_train = label_train.ravel()
        label_val = label_val.ravel()
        C_cands = 10. ** np.arange(-3, 1, 0.5)
        print('C candidates: ', C_cands)
        acc_c=[]
        val_acc_best=0
        best_C=None
        for C in C_cands:
            clf = LinearSVC(random_state=7, C=C).fit(data_train, label_train)
            preds_train = clf.predict(data_train)
            preds_val = clf.predict(data_val)
            train_acc = np.sum(preds_train == label_train) / len(label_train)
            val_acc = np.sum(preds_val == label_val) / len(label_val)
            if val_acc > val_acc_best:
                val_acc_best = val_acc
                best_C = C
                print(best_C)
                best_fold=fold
                model_save_path = os.path.join(save_dir, 'svm_fold_{}.joblib'.format(fold))
                joblib.dump(clf, model_save_path)
            print('C:', C, 'train acc:', train_acc, 'val acc:', val_acc)
            acc_c.append((train_acc,val_acc))
        c_best = max(acc_c, key=lambda x: x[1])
        val_acc_list.append(c_best[1])
        train_acc_list.append(c_best[0])

        print('Best C for fold', best_fold, ':', best_C)
        print('Best validation accuracy for fold', best_fold, ':', val_acc_best)
        if fold==9:
            print('Average Train Accuracy:', np.mean(train_acc_list))
            print('Average Validation Accuracy:', np.mean(val_acc_list))
    return best_fold
def evaluate(subject_metrics,test_data_subject, test_labels_subject, clf2):
    predictions_subject = clf2.predict(test_data_subject)
    accuracy_subject = accuracy_score(test_labels_subject, predictions_subject)
    f1_micro = f1_score(test_labels_subject, predictions_subject, average='micro')
    f1_macro = f1_score(test_labels_subject, predictions_subject, average='macro')
    precision_micro = precision_score(test_labels_subject, predictions_subject, average='micro')
    precision_macro = precision_score(test_labels_subject, predictions_subject, average='macro')
    recall_micro = recall_score(test_labels_subject, predictions_subject, average='micro')
    recall_macro = recall_score(test_labels_subject, predictions_subject, average='macro')
    kappa = cohen_kappa_score(test_labels_subject, predictions_subject)
    subject_metrics.append({
    'accuracy': accuracy_subject,
    'f1_micro': f1_micro,
    'f1_macro': f1_macro,
    'precision_micro': precision_micro,
    'precision_macro': precision_macro,
    'recall_micro': recall_micro,
    'recall_macro': recall_macro,
    'kappa': kappa
})
    return subject_metrics
def evaluate_subjects(fold_subject_metrics,test_indices, test_data, test_labels, clf2):
    subject_metrics = []
    for idx, (start_index, end_index) in enumerate(test_indices):
        test_data_subject = test_data[start_index:end_index]
        test_labels_subject = test_labels[start_index:end_index]
        subject_metrics= evaluate(subject_metrics,test_data_subject, test_labels_subject, clf2)
    fold_subject_metrics.append(subject_metrics)
    return fold_subject_metrics, subject_metrics
def evaluate_kfold(subject_metrics,test_data, test_labels, average_cm , clf2):
    predictions_subject = clf2.predict(test_data)
    subject_cm = confusion_matrix(test_labels, predictions_subject,normalize='true')
    subject_metrics = evaluate(subject_metrics,test_data, test_labels, clf2)
    if subject_cm.shape == (3, 3):
        new_cm = np.zeros((4, 4))
        new_cm[:3, :3] = subject_cm
        subject_cm=new_cm
    if subject_cm.shape == (2, 2):
        new_cm = np.zeros((4, 4))
        new_cm[:2, :2] = subject_cm
        subject_cm=new_cm
    if average_cm is None:
        average_cm = [subject_cm ]
    else:
        average_cm.append(subject_cm)
    return subject_metrics,average_cm
def calculate_folds_statistics(subject_metrics):
    metrics_values = {key: [] for key in subject_metrics[0].keys() if key != 'confusion_matrix'}
    for metrics in subject_metrics:
        for key in metrics_values:
            metrics_values[key].append(metrics[key])
    metrics_statistics = {}
    for key, values in metrics_values.items():
        mean = np.mean(values)
        std_error = np.std(values, ddof=1) / np.sqrt(len(values))
        metrics_statistics[key] = {'mean': mean, 'std_error': std_error}
    return metrics_statistics
def calculate_subject_avg_metrics(subject_metrics):
    avg_subject_metrics = []
    std_error_subject_metrics = []
    num_subjects = len(subject_metrics[0])
    for i in range(num_subjects):
        subject_avg_metrics = {}
        subject_values = {}
        num_folds = len(subject_metrics)
        for fold_metrics in subject_metrics:
            for metric_name, metric_value in fold_metrics[i].items():
                if metric_name not in subject_avg_metrics:
                    subject_avg_metrics[metric_name] = 0
                    subject_values[metric_name] = []
                subject_avg_metrics[metric_name] += metric_value / num_folds
                subject_values[metric_name].append(metric_value)

        avg_subject_metrics.append(subject_avg_metrics)
        std_error_subject = {}
        for metric_name, values in subject_values.items():
            std_error_subject[metric_name] = np.std(values, ddof=1) / np.sqrt(num_folds)
        std_error_subject_metrics.append(std_error_subject)
    return avg_subject_metrics, std_error_subject_metrics

warnings.filterwarnings("ignore", category=FutureWarning)
np.random.seed(7)

data_path = './derivatives_data/feature/mat'
# data_path = './your_feature_path'
data_paths = os.listdir(data_path)
# type = '4class'
type = '2class'

test_indices = []
n_folds = 8
folds_list = np.arange(0, n_folds)
n_subs = 23
best_fold=None
best_C = None

if type == '2class':
    save_dir = './derivatives_data/code/Classification_validation/svm_model/2class'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    label_map = {1001: 0, 1002: 1, 1004: 1, 1007: 1}
if type == '4class':
    save_dir = './derivatives_data/code/Classification_validation/svm_model/4class'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    label_map = {1001: 0, 1002: 1, 1004: 2, 1007: 3}
data = None
labels = None
test_labels = None
test_data = None
average_cm = None
kfold_metrics = []
fold_subject_metrics = []
fold_subject_cm = []
train_acc_list = []
val_acc_list = []
val_acc_best = 0
test_indices = []
for idx, file in enumerate(data_paths):
    s = get_all_files(os.path.join(data_path, file))
    datafiles = sorted(s)
    for datafile in datafiles:
        if 'sub_label' in datafile:
            label_sub = np.load(datafile)
            # print(f"Label raw shape:", label_sub.shape)
        elif 'sub_psd' in datafile:
            raw = np.load(datafile)
            # print(f"Raw shape:", raw.shape)
            data_sub = raw
    class_counts = Counter(label_sub.flatten().tolist())
    for class_label, count in class_counts.items():
        continue
    new_label_sub = None
    new_data_sub = None
    for i in range(label_sub.shape[0]):
        p = data_sub[i].reshape(1, data_sub[i].shape[0], data_sub[i].shape[1])
        q = label_sub[i]
        if isinstance(q, (int, np.int64)) or isinstance(q, str):
            q = [q]
        e = q[0]
        if class_counts[e] < n_folds:
            continue
        else:
            if np.array_equal(new_label_sub, None):
                new_label_sub = q
                new_data_sub = p
            else:
                new_label_sub = np.concatenate((new_label_sub, q), axis=0)
                new_data_sub = np.concatenate((new_data_sub, p), axis=0)
    label_sub = new_label_sub
    # print(label_sub.shape)
    data_sub = new_data_sub
    X_train, X_test, y_train, y_test = train_test_split(data_sub, label_sub, test_size=0.1, random_state=42,
                                                        stratify=label_sub)
    # print(f"y_test shape:", y_test.shape)
    if idx == 0:
        data = X_train
        labels = y_train
        test_data = X_test
        test_labels = y_test
        start_index = 0
    else:
        data = np.concatenate((data, X_train), axis=0)
        labels = np.concatenate((labels, y_train), axis=0)
        test_data = np.concatenate((test_data, X_test), axis=0)
        test_labels = np.concatenate((test_labels, y_test), axis=0)
        start_index = test_indices[-1][1]
    end_index = start_index + len(y_test)
    test_indices.append((start_index, end_index))
# print(f"test_data:", test_data.shape)
# print(f"test_labels:", test_labels.shape)

data, labels  = preprocess_test_data(data,labels,label_map)
best_fold = train_svm(best_fold,data, labels)
test_data, test_labels = preprocess_test_data(test_data, test_labels, label_map)
for fold in folds_list:
    model_path = os.path.join(save_dir, 'svm_fold_{}.joblib'.format(fold))
    clf2 = joblib.load(model_path)
    kfold_metrics, average_cm = evaluate_kfold(kfold_metrics, test_data, test_labels, average_cm, clf2)
    fold_subject_metrics, subject_metrics = evaluate_subjects(fold_subject_metrics, test_indices, test_data,
                                                              test_labels, clf2)


mean_cm = np.mean(np.stack(average_cm), axis=0)
average_cm_save_path = os.path.join(save_dir, 'average_cm.npy')
np.save(average_cm_save_path, average_cm)
save_path = os.path.join(save_dir, 'mean_cm.npy')
np.save(save_path, mean_cm)

folds_statistics = calculate_folds_statistics(kfold_metrics)
print(folds_statistics)
folds_statistics_save_path = os.path.join(save_dir, 'intra_folds_statistics .npy')
np.save(folds_statistics_save_path, folds_statistics)

avg_metrics, std_error_metrics = calculate_subject_avg_metrics(fold_subject_metrics)
print("Average metrics:", avg_metrics)
print("Standard error:", std_error_metrics)
accuracies = [entry['accuracy'] for entry in avg_metrics]
print("acc:", accuracies)
save_path0 = os.path.join(save_dir, 'avg_metrics.npy')
np.save(save_path0, avg_metrics)
save_path1 = os.path.join(save_dir, 'std_error_metrics.npy')
np.save(save_path1, std_error_metrics)


