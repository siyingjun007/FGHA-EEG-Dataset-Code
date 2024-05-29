import numpy as np
import pandas as pd
import h5py
import os
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns

fig = plt.figure(figsize=(10, 5), dpi=500)
plt.rcParams.update({'font.family': 'Arial','font.size': 12, 'font.weight': 'normal'})
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

ax2 = fig.add_subplot(gs[0, 0])
def get_all_files(directory):
    all_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files
# data_path = 'your_feature_path'
data_path = './derivatives_data/feature/mat'
data_paths = os.listdir(data_path)
data = []
labels = []
full_paths = [os.path.join(data_path, path) for path in data_paths]
person_class_counts = {}
def sort_by_number(file_name):
    return int(file_name.split('-')[1])
sorted_data_paths = sorted(data_paths, key=sort_by_number)
for idx, file in enumerate(sorted_data_paths):
    s = get_all_files(os.path.join(data_path, file))
    datafiles = sorted(s)
    data_sub = None
    label_sub = None
    for datafile in datafiles:
        if 'label' in datafile and 'sub_label' not in datafile:
            with h5py.File(datafile, 'r') as f:
                labelraw = f['label'][:]
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
            if data_sub is None:
                data_sub = raw
            else:
                data_sub = np.concatenate((data_sub, raw), axis=0)
    class_counts = Counter(label_sub.flatten().tolist())
    person_class_counts[idx] = class_counts
    if idx==0:
        labels = label_sub
    else:
        labels = np.concatenate((labels, label_sub), axis=0)
df = pd.DataFrame(columns=range(1, 5), index=range(1, 23))
for idx, class_counts in person_class_counts.items():
    for class_label, count in class_counts.items():
        df.at[idx, class_label] = count
df.fillna(0, inplace=True)
interested_columns = [1001, 1002, 1004, 1007]
interested_df = df[interested_columns]
data_array = interested_df.values
data_array = np.roll(data_array, 1, axis=0)
cmap = plt.colormaps["tab10"]
inner_colors = cmap([1, 2, 3, 4, 5, 6,7,8])

complexity_levels = ['Level 0', 'Level 1', 'Level 2', 'Level 3']
x_labels=complexity_levels
label_counts = Counter(labels.flatten().tolist())
sample_sizes = [label_counts.get(label, 0) for label in interested_columns]
print(f'sample_sizes = {sample_sizes}')
# sample_sizes = [2118, 3472, 1628, 567]
bars = ax2.bar(complexity_levels, sample_sizes,color=cmap(0),alpha=0.3)
ax2.set_ylabel('Total Epochs', fontsize=14)
ax2.set_xlabel('Hypoxia Levels', fontsize=14)
ax2.set_ylim(0, max(sample_sizes) + 500)
for bar in bars:
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
             f'{bar.get_height()}', ha='center', va='bottom', fontsize=12)
ax1=ax2.twinx()
y=np.arange(4)
x=[3,4,7,16,17,19,20,21]
jitter_strength = 0.2
j=0
for i in x:
    non_zero_indices = np.where(data_array[i] != 0)[0]
    non_zero_values = data_array[i][non_zero_indices]
    if non_zero_values.size > 0:
        jittered_y = y[non_zero_indices] + np.random.rand(non_zero_indices.size) * jitter_strength - (jitter_strength / 2)
        ax1.scatter(jittered_y, non_zero_values, label=f'Sub-{i+1}',c=inner_colors[j])
    j=j+1
ax1.set_ylim(0, 285)
ax1.set_ylabel('Epochs for a Single Participant', fontsize=14)
ax1.legend(loc='upper right',frameon=False)
plt.rcParams.update({'axes.linewidth': 1, 'axes.edgecolor': 'black'})
plt.xlabel('Hypoxia Levels', fontsize=14)

ax3 = fig.add_subplot(gs[0, 1])


psd_df = pd.DataFrame(columns=['psd_level0', 'psd_level1', 'psd_level2', 'psd_level3'])
for i in range(4):
    # data_path = 'your_feature_psd_path'
    data_path = './derivatives_data/feature/psd'
    file = os.path.join(data_path, f'psd_level{i}.npy')
    psd_data = np.load(file)
    psd_data_flat = psd_data.flatten()
    psd_df[f'psd_level{i}'] = psd_data_flat

cmap = plt.colormaps["tab10"]
inner_colors = cmap([0,1, 2, 3])
palette=inner_colors
sns.barplot(data=psd_df, palette=palette,capsize=.1,errwidth=1,errcolor="k",**{"edgecolor":"k","linewidth":1})
plt.xlabel('Hypoxia Levels', fontsize=14)
plt.ylabel('PSD Value', fontsize=14)
plt.ylim(0,0.356)
plt.xticks(ticks=range(4), labels=["Level 0", "Level 1", "Level 2", "Level 3"])
legend_labels = ['Level 0', 'Level 1', 'Level 2', 'Level 3']
legend_handles = [plt.Rectangle((0,0),1,1, color=color) for color in inner_colors]
plt.legend(handles=legend_handles, labels=legend_labels, loc='upper center', bbox_to_anchor=(0.5, 1), fontsize=12 ,frameon=False,ncol=2)
fig.text(0.01, 0.99, "a", size=14, rotation=0, ha="left", va="top", bbox=dict(boxstyle="square", ec='white', fc='white', alpha=0.1))
fig.text(0.54, 0.99, "b", size=14, rotation=0, ha="left", va="top", bbox=dict(boxstyle="square", ec='white', fc='white', alpha=0.1))
plt.tight_layout()
plt.savefig('avn.png', dpi=600, pad_inches=0)
plt.savefig('avn.pdf', pad_inches=0)
plt.show()