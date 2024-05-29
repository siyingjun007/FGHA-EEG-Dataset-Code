import os
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from keras import Sequential
from keras.layers import DepthwiseConv1D, Conv1D, Activation, Flatten, Dense, Dropout, BatchNormalization, Input, AveragePooling1D, SeparableConv1D
from keras.models import Model
from keras.callbacks import Callback
from keras.constraints import max_norm
from keras.losses import categorical_crossentropy
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, cohen_kappa_score
from collections import Counter
import matplotlib.pylab as plt

def get_all_files(directory):
    all_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files
def design_model():
    inp = Input(shape=(n_channnel, n_time))
    print(inp.shape)
    conv1 = Conv1D(64, 3, padding='same', use_bias=False)(inp)
    conv1 = DepthwiseConv1D(64, padding='same', use_bias=False, depth_multiplier=1,
                            depthwise_constraint=max_norm(1.))(conv1)
    conv1 = BatchNormalization()(conv1)
    l1 = Activation('elu')(conv1)
    l2 = AveragePooling1D(2)(l1)
    # print(l2.shape)
    dl2 = Dropout(0.5)(l2)
    conv2 = SeparableConv1D(32, 32, padding='same', use_bias=False)(dl2)
    conv2 = BatchNormalization()(conv2)
    l3 = Activation('elu')(conv2)
    l3 = AveragePooling1D(2)(l3)
    m3 = Dropout(0.5)(l3)
    f = Flatten()(m3)
    result = Dense(n_class, kernel_constraint=max_norm(0.25), activation='softmax')(f)
    model = Model(inputs=inp, outputs=result)
    model.summary()
    adam = keras.optimizers.Adam(learning_rate=lr, beta_1=0.95, beta_2=0.999, epsilon=1e-08)
    model.compile(loss=myloss, optimizer=adam, metrics=['accuracy'])
    return model
def combined_dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return 1 - score
def myloss(y_true, y_pred, e=0.9):
    loss1 = categorical_crossentropy(y_true, y_pred)
    loss2 = combined_dice_loss(y_true, y_pred, smooth=1e-6)
    return (1 - e) * loss1 + e * loss2
class CustomModelCheckpoint(Callback):
    def __init__(self, save_dirr, base_filename, fold, period, monitor='val_loss', mode='min', save_best_only=True):
        super(CustomModelCheckpoint, self).__init__()
        self.save_dirr = save_dirr
        self.base_filename = base_filename
        self.fold = fold
        self.period = period
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best = np.Inf if mode == 'min' else -np.Inf
    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if self.mode == 'min' and current < self.best:
            self.best = current
            is_best = True
        elif self.mode == 'max' and current > self.best:
            self.best = current
            is_best = True
        else:
            is_best = False
        if epoch % self.period == 0 or is_best:
            filename = self.base_filename.format(epoch=epoch, val_accuracy=logs.get('val_accuracy'))
            filepath = os.path.join(self.save_dirr, f'fold{self.fold}', filename)
            if not os.path.exists(os.path.dirname(filepath)):
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
            self.model.save_weights(filepath, overwrite=True)
def train_model(best_fold_metrics,data, labels):
    scores = []
    print(f" data shape:", data.shape)
    labels = np.vectorize(label_map.get)(labels)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    fold = -1
    for train_index, test_index in skf.split(data, labels):
        fold = fold+1
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test =labels[train_index], labels[test_index]
        print('fold', fold)
        data_train=X_train
        data_val= X_test
        label_train=y_train
        label_val=y_test
        # print('data_train: ', data_train.shape)
        # print('data_val: ', data_val.shape)
        label_train = label_train.ravel()
        label_val = label_val.ravel()
        label_train = to_categorical(label_train, nb_classes)
        label_val = to_categorical(label_val, nb_classes)
        # print('label_train: ', label_train.shape)
        # print('label_val: ', label_val.shape)
        model = design_model()
        base_dir1 = '{fold}'.format(fold=fold)
        save_dirr = os.path.join(save_dir, base_dir1)
        base_filename = 'model_{epoch:02d}-{val_accuracy:.2f}.weights.h5'

        if not os.path.exists(save_dirr):
            os.makedirs(save_dirr, exist_ok=True)
        current_fold = fold
        custom_checkpoint = CustomModelCheckpoint(save_dirr=save_dirr, base_filename=base_filename,
                                                  fold=current_fold, period=50, monitor='val_loss', mode='min',
                                                  save_best_only=True)
        network_history = model.fit(data_train, label_train, epochs=epochs, batch_size=batch_size, shuffle=True,
                                    verbose=0,
                                    validation_data=(data_val, label_val), callbacks=[custom_checkpoint])
        best_val_accuracy = max(network_history.history["val_accuracy"])
        best_epoch = network_history.history["val_accuracy"].index(best_val_accuracy)
        best_fold_metrics.append((fold, best_val_accuracy, best_epoch))
        scores.append(
            (network_history.history["loss"], network_history.history["accuracy"], network_history.history["val_loss"],
             network_history.history["val_accuracy"]))
        scoresmean = np.mean(scores, axis=0)
        base_dir = '{fold}/scoresmean.npy'.format(fold=fold)
        scoressave_path = os.path.join(save_dir, base_dir)
        np.save(scoressave_path, scoresmean)
    labs = ["loss", "accuracy", "val_loss", "val_accuracy"]
    result = zip([l + '_mean' for l in labs], [s[-1] for s in scoresmean])
    [print(res) for res in result]
    color = ['black', 'black', 'red', 'red']
    [plt.plot(scoresmean[i], label=labs[i], color=color[i]) for i in range(len(scoresmean))]
    plt.legend()
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss/Accuracy', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('model 10KLoss/Accuracy')
    base_dir1 = '{fold}/10Kimage.png'.format(fold=fold)
    picpath = os.path.join(save_dir, base_dir1)
    plt.savefig(picpath, dpi=500, bbox_inches='tight')
    plt.close()
    return best_fold_metrics
def find_best_model(save_dir,fold) :
    base_dir1 = str(fold)
    save_dirr = os.path.join(save_dir, base_dir1)
    fname = f"fold{fold}"
    save_dirr = os.path.join(save_dirr, fname)
    print(save_dirr)
    files = os.listdir(save_dirr)
    third_elements = []
    for name in files:
        start = name.rfind('-') + 1
        end = name.rfind('.weights')
        accuracy = float(name[start:end])
        third_elements.append(accuracy)
    max_third_index = third_elements.index(max(third_elements))
    max_third_file_name = files[max_third_index]
    print(max_third_file_name)
    second_elements = [int(name.split('_')[1].split('-')[0]) for name in files if float(name[name.rfind('-') + 1:name.rfind('.weights')]) == max(third_elements)]
    max_second_index = second_elements.index(max(second_elements))
    max_second_file_name = [name for name in files if float(name[name.rfind('-') + 1:name.rfind('.weights')]) == max(third_elements)][max_second_index]
    print("Filename with the highest third element:", max_third_file_name)
    print("Filename with the highest second element among those with the highest third element:", max_second_file_name)
    return save_dirr, max_second_file_name

def evaluate(subject_metrics, test_data_subject, test_labels_subject, clf2):
    predictions_subject = clf2.predict(test_data_subject)
    predictions = np.argmax(predictions_subject, axis=1)
    accuracy_subject = accuracy_score(test_labels_subject, predictions)
    f1_micro = f1_score(test_labels_subject, predictions, average='micro')
    f1_macro = f1_score(test_labels_subject, predictions, average='macro')
    precision_micro = precision_score(test_labels_subject, predictions, average='micro')
    precision_macro = precision_score(test_labels_subject, predictions, average='macro')
    recall_micro = recall_score(test_labels_subject, predictions, average='micro')
    recall_macro = recall_score(test_labels_subject, predictions, average='macro')
    kappa = cohen_kappa_score(test_labels_subject, predictions)
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
def evaluate_subjects(fold_subject_metrics, test_indices, test_data, test_labels, clf2):
    subject_metrics = []
    for idx, (start_index, end_index) in enumerate(test_indices):
        test_data_subject = test_data[start_index:end_index]
        test_labels_subject = test_labels[start_index:end_index]
        subject_metrics = evaluate(subject_metrics, test_data_subject, test_labels_subject, clf2)
    fold_subject_metrics.append(subject_metrics)
    return fold_subject_metrics, subject_metrics
def evaluate_kfold(subject_metrics, test_data, test_labels, average_cm, clf2):
    predictions_subject = clf2.predict(test_data)
    predictions = np.argmax(predictions_subject, axis=1)
    subject_cm = confusion_matrix(test_labels, predictions, normalize='true')
    subject_metrics = evaluate(subject_metrics, test_data, test_labels, clf2)
    if subject_cm.shape == (3, 3):
        new_cm = np.zeros((4, 4))
        new_cm[:3, :3] = subject_cm
        subject_cm = new_cm
    if subject_cm.shape == (2, 2):
        new_cm = np.zeros((4, 4))
        new_cm[:2, :2] = subject_cm
        subject_cm = new_cm
    if average_cm is None:
        average_cm = [subject_cm]
    else:
        average_cm.append(subject_cm)
    return subject_metrics, average_cm
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

data_path = './derivatives_data/feature/mat'
# data_path = './your_feature_path'
data_paths = os.listdir(data_path)
# type='2class'
type='4class'

if type=='2class':
    save_dir = './derivatives_data/code/Classification_validation/EEGnetmodel/2class'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    n_class = 2
    label_map = {1001: 0, 1002: 1, 1004: 1, 1007: 1}
    n_time = 251
    n_channnel = 61
    nb_classes = n_class
    epochs = 651
    lr = 0.0001
    scores = []
    batch_size = 128
if type=='4class':
    save_dir = './derivatives_data/code/Classification_validation/EEGnetmodel/4class'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    n_class = 4
    label_map = {1001: 0, 1002: 1, 1004: 2, 1007: 3}
    n_time = 251
    n_channnel = 61
    nb_classes = n_class
    epochs = 801
    lr = 0.0001
    scores = []
    batch_size = 128
np.random.seed(7)
n_folds = 10
folds_list = np.arange(0, n_folds)
n_subs = 23
n_per = round(n_subs / n_folds)

data = None
labels = None
test_labels = None
test_data = None
t = 0
best_fold_metrics = []
test_indices = []
fold_subject_metrics = []
fold_subject_cm=[]
kfold_metrics=[]
average_cm=None

for idx, file in enumerate(data_paths):
    s = get_all_files(os.path.join(data_path, file))
    datafiles = sorted(s)
    for datafile in datafiles:
        if 'sub_label' in datafile:
            label_sub = np.load(datafile)
            label_sub = label_sub.astype(int)
            # print(f"Label raw shape:", label_sub.shape)
        elif 'sub_psd' in datafile:
            raw = np.load(datafile)
            #  print(f"Raw shape:", raw.shape)
            data_sub = raw
    class_counts = Counter(label_sub.flatten().tolist())
    for class_label, count in class_counts.items():
        # print(f"Class {class_label}: {count} samples")
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
test_labels = np.vectorize(label_map.get)(test_labels)
print(f"test_data:", test_data.shape)
print(f"test_labels:", test_labels.shape)
indicessave_path = os.path.join(save_dir, 'test_indices.npy')
np.save(indicessave_path, test_indices)
best_fold_metrics = train_model(best_fold_metrics,data,labels)

for fold in folds_list:
    modeldir,best_model_filename = find_best_model(save_dir, fold)
    print("The model file name corresponding to the highest accuracy:", best_model_filename)
    intramodel_path = os.path.join( modeldir, best_model_filename)
    model=design_model()
    model.load_weights(intramodel_path)
    model.summary()
    kfold_metrics, average_cm = evaluate_kfold(kfold_metrics,test_data, test_labels, average_cm,model)
    fold_subject_metrics,subject_metrics = evaluate_subjects(fold_subject_metrics,test_indices,test_data,test_labels, model)

mean_cm = np.mean(np.stack(average_cm), axis=0)
save_path= os.path.join(save_dir,'mean_cm.npy')
np.save(save_path, mean_cm)

folds_statistics = calculate_folds_statistics(kfold_metrics)
print(folds_statistics)
folds_statistics_save_path = os.path.join(save_dir, 'intra_folds_statistics .npy')
np.save(folds_statistics_save_path, folds_statistics)

avg_metrics, std_error_metrics = calculate_subject_avg_metrics(fold_subject_metrics)
print("Average metrics:", avg_metrics)
accuracies = [entry['accuracy'] for entry in avg_metrics]
print("accuracy:", accuracies)
accuracies_save_path = os.path.join(save_dir, 'intra_avgaccuracies.npy')
np.save(accuracies_save_path, accuracies)
print("Standard error:", std_error_metrics)
save_path0 = os.path.join(save_dir, 'avg_metrics.npy')
np.save(save_path0, avg_metrics)
save_path1 = os.path.join(save_dir, 'std_error_metrics.npy')
np.save(save_path1, std_error_metrics)
