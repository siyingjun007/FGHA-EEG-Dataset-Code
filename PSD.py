import os
import numpy as np
import h5py
from scipy import signal
import mne

def get_all_files(directory):
    all_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files

# data_path = 'your_mat_EEG_data_path'
data_path = './derivatives_data/feature/mat'
data_paths = os.listdir(data_path)
data = []
labels = []
full_paths = [os.path.join(data_path, path) for path in data_paths]
# print(full_paths)
for idx, file in enumerate(data_paths):
    s = get_all_files(os.path.join(data_path, file))
    datafiles = sorted(s)
    data_sub = None
    label_sub = None
    for datafile in datafiles:
        if 'label' in datafile and 'sub_label' not in datafile:
            with h5py.File(datafile, 'r') as f:
                labelraw = f['label'][:]
                #print(f"Label raw shape:", labelraw.shape)
            if label_sub is None:
                label_sub = labelraw
            else:
                label_sub = np.concatenate((label_sub, labelraw), axis=0)
        elif datafile.endswith('.mat'):
            with h5py.File(datafile, 'r') as f:
                raw = f['EEGtemp']['data'][:]
                if raw.ndim == 2:
                    raw = np.expand_dims(raw, axis=2)
                    raw = raw.transpose(2, 0, 1)
                if raw.ndim == 3:
                    raw = raw.transpose(0, 2, 1)
                    #print(f"Raw shape:", raw.shape)
                _, psd_delta = signal.welch(raw, fs=500, nperseg=500)
                # print(f"psd_delta  shape:", psd_delta .shape)
            if data_sub is None:
                data_sub = raw
                data_subpsd= psd_delta
            else:
                data_sub = np.concatenate((data_sub, raw), axis=0)
                data_subpsd = np.concatenate((data_subpsd, psd_delta), axis=0)
    #print(f"Label sub shape:", label_sub.shape)
    #print(f" data_subpsd shape:",  data_subpsd.shape)
    output_dir = full_paths[idx]
    filename = "sub_psd.npy"
    output_file = os.path.join(output_dir, filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.save(output_file, data_subpsd)
    filename = "sub_label.npy"
    output_file = os.path.join(output_dir, filename)
    np.save(output_file, label_sub)
    if idx==0:
        labels = label_sub
        data = data_sub
    else:
        labels = np.concatenate((labels, label_sub), axis=0)
        data = np.concatenate((data, data_sub), axis=0)

sampling_freq=500
ch_names =['Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'POz', 'O1', 'O2', 'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CP4', 'P5', 'P1', 'P2', 'P6', 'PO5', 'PO3', 'PO4', 'PO6', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8', 'Oz']
ch_types = ['eeg']*61
info=mne.create_info(ch_names,ch_types='eeg',sfreq=sampling_freq)
info.set_montage('standard_1020')
info['description'] = 'Hypoxic data'
print(data.shape)
samplsnum=data.shape[2]
labelnum=labels.shape[0]
print(labels.shape)
event = np.arange(1501,labels.shape[0]*samplsnum,samplsnum)
events = np.column_stack([event.astype(int),np.zeros(labels.shape[0],dtype=int), labels])
# print(events.shape)
event_dict = {'0':1001,'1': 1002,'2': 1004,'3': 1007}
events = mne.pick_events(events, include=[1001,1002,1004,1007])
events = np.array(events)
data_sub=None
raw=None
datafiles=None
data_paths=None
labels=None
label_sub=None
labelraw=None
event=None
s=None

psd_save_path='./derivatives_data/feature/psd'
if not os.path.exists(psd_save_path):
    os.makedirs(psd_save_path)
simulated_epochs = mne.EpochsArray(data,info,tmin=-3,events=events.astype(int),event_id=event_dict)
data=None
events=None
_, psd3 = signal.welch(simulated_epochs['3'], fs=500, nperseg=500)
psd_cond3= np.mean(psd3, axis=0)
psd3=None
filename= 'psd_level3.npy'
output_file = os.path.join(psd_save_path, filename)
np.save(output_file, psd_cond3)
_, psd2 = signal.welch(simulated_epochs['2'], fs=500, nperseg=500)
psd_cond2 = np.mean(psd2, axis=0)
psd2=None
filename= 'psd_level2.npy'
output_file = os.path.join(psd_save_path, filename)
np.save(output_file, psd_cond2)
_, psd1 = signal.welch(simulated_epochs['1'], fs=500, nperseg=500)
psd_cond1= np.mean(psd1, axis=0)
psd1=None
filename= 'psd_level1.npy'
output_file = os.path.join(psd_save_path, filename)
np.save(output_file, psd_cond1)
_, psd0 = signal.welch(simulated_epochs['0'], fs=500, nperseg=500)
psd_cond0= np.mean(psd0, axis=0)
psd0=None
filename= 'psd_level0.npy'
output_file = os.path.join(psd_save_path, filename)
np.save(output_file, psd_cond0)