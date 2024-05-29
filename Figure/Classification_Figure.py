import seaborn as sns
import numpy as np
from matplotlib import gridspec
import matplotlib.pyplot as plt
cmap = plt.colormaps["tab10"]
inner_colors = cmap([1, 2, 3, 4, 5, 6,7,8])
plt.figure(figsize=(12, 8), dpi=500)
plt.rcParams.update({'font.family': 'Arial','font.size': 12, 'font.weight': 'normal'})
type = '2class'
# type = '4class'
if type == '2class':
    mean_cm1 = np.load('./derivatives_data/code/Classification_validation/svm_model/2class/mean_cm.npy')
    mean_cm1 = mean_cm1[:2, :2]
    accuracies = np.load('./derivatives_data/code/Classification_validation/svm_model/2class/avg_metrics.npy',
                         allow_pickle=True)
    accuracies1 = [entry['accuracy'] for entry in accuracies]
    mean_cm2 = np.load(
        './derivatives_data/code/Classification_validation/EEGnet_model/2class/mean_cm.npy')
    mean_cm2 = mean_cm2[:2, :2]
    accuracies2 = np.load(
        './derivatives_data/code/Classification_validation/EEGnet_model/2class/intra_avgaccuracies.npy')
    gs = gridspec.GridSpec(2, 2, width_ratios=[2.1, 1])

    ax0 = plt.subplot(gs[1, 1])
    ax0.text(-0.68, -0.3, "d", size=14, rotation=0, ha="left", va="top",
             bbox=dict(boxstyle="square", ec='white', fc='white', alpha=0.1))
    sns.heatmap(mean_cm2, annot=True, cmap="Blues", fmt=".2f",
                xticklabels=["Level 0", "Level 1"],
                yticklabels=["Level 0", "Level 1"])
    ax0.set_xticklabels(ax0.get_xticklabels(), rotation=45, ha='right')
    ax0.set_yticklabels(ax0.get_yticklabels(), rotation=45, ha='right')
    plt.xlabel("Predicted Label", fontsize=14)
    plt.ylabel("True Label", fontsize=14)
    ax1 = plt.subplot(gs[1, 0])
    ax1.text(-1.6, 1.065, "b", size=14, rotation=0, ha="left", va="top",
             bbox=dict(boxstyle="square", ec='white', fc='white', alpha=0.1))
    xlabels = range(1, 24, 1)
    plt.bar(xlabels, accuracies2, color=cmap(0), label='Accuracy')
    mean_accuracy = np.mean(accuracies2)
    plt.axhline(y=mean_accuracy, color='black', linestyle='--', label='Mean Accuracy')
    plt.xlabel('Subjects', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.xlim(0.5, 23.5)
    plt.ylim(0.6, 1.005)
    plt.xticks(xlabels)
    legend = plt.legend(loc='lower right')
    legend.get_frame().set_alpha(0.95)
    plt.text(0.3, mean_accuracy, f'{mean_accuracy:.2f}', ha='right', va='center', color='black', fontsize=12)
    ax3 = plt.subplot(gs[0, 1])
    ax3.text(-0.68, -0.3, "c", size=14, rotation=0, ha="left", va="top",
             bbox=dict(boxstyle="square", ec='white', fc='white', alpha=0.1))
    ax3 = sns.heatmap(mean_cm1, annot=True, cmap="Blues", fmt=".2f",
                      xticklabels=[],
                      yticklabels=["Level 0", "Level 1"])
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
    ax3.set_yticklabels(ax3.get_yticklabels(), rotation=45, ha='right')
    plt.ylabel("True Label", fontsize=14)

    ax4 = plt.subplot(gs[0, 0])
    plt.text(-1.6, 1.065, "a", size=14, rotation=0, ha="left", va="top",
             bbox=dict(boxstyle="square", ec='white', fc='white', alpha=0.1))
    plt.bar(xlabels, accuracies1, color=cmap(0), label='Accuracy')
    mean_accuracy = np.mean(accuracies1)
    plt.axhline(y=mean_accuracy, color='black', linestyle='--', label='Mean Accuracy')
    plt.ylabel('Accuracy', fontsize=14)
    plt.xlim(0.5, 23.5)
    plt.ylim(0.6, 1.005)
    plt.xticks(xlabels)
    legend = plt.legend(loc='lower right')
    legend.get_frame().set_alpha(0.95)
    plt.text(0.3, mean_accuracy, f'{mean_accuracy:.2f}', ha='right', va='center', color='black', fontsize=12)
    plt.tight_layout()
    plt.savefig('2class.png', dpi=500, pad_inches=0)
    plt.savefig('2class.pdf', pad_inches=0)

if type=='4class':
    mean_cm1=np.load('./derivatives_data/code/Classification_validation/svm_model/4class/mean_cm.npy')
    accuracies=np.load('./derivatives_data/code/Classification_validation/svm_model/4class/avg_metrics.npy',allow_pickle=True)
    accuracies1= [entry['accuracy'] for entry in accuracies]
    mean_cm2=np.load('./derivatives_data/code/Classification_validation/EEGnet_model/4class/mean_cm.npy')
    accuracies2=np.load('./derivatives_data/code/Classification_validation/EEGnet_model/4class/intra_avgaccuracies.npy')
    gs = gridspec.GridSpec(2, 2, width_ratios=[2.1, 1])

    ax0 = plt.subplot(gs[1, 1])
    ax0.text( -1.3,-0.59,"d",size=14,rotation=0,ha="left",va="top",bbox=dict(boxstyle="square",ec='white',fc='white',alpha=0.1))
    sns.heatmap(mean_cm2, annot=True, cmap="Blues", fmt=".2f",
                xticklabels=["Level 0", "Level 1", "Level 2", "Level 3"],
                yticklabels=["Level 0", "Level 1", "Level 2", "Level 3"])
    ax0.set_xticklabels(ax0.get_xticklabels(), rotation=45, ha='right')
    ax0.set_yticklabels(ax0.get_yticklabels(), rotation=45, ha='right')
    plt.xlabel("Predicted Label", fontsize=14)
    plt.ylabel("True Label", fontsize=14)
    ax1 = plt.subplot(gs[1, 0])
    ax1.text(-1.6,1.06,"b",size=14,rotation=0,ha="left",va="top",bbox=dict(boxstyle="square",ec='white',fc='white',alpha=0.1))
    xlabels = range(1, 24,1)
    plt.bar(xlabels, accuracies2, color=cmap(0), label='Accuracy')
    mean_accuracy= np.mean(accuracies2)
    plt.axhline(y=mean_accuracy, color='black', linestyle='--', label='Mean Accuracy')
    plt.xlabel('Subjects', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.xlim(0.5, 23.5)
    plt.ylim(0.6, 1.005)
    plt.xticks(xlabels)
    legend = plt.legend(loc='lower right')
    legend.get_frame().set_alpha(0.95)
    plt.text(0.3, mean_accuracy, f'{mean_accuracy:.2f}', ha='right', va='center', color='black', fontsize=12)

    ax3 = plt.subplot(gs[0, 1])
    ax3.text(-1.3,-0.5,"c",size=14,rotation=0,ha="left",va="top",bbox=dict(boxstyle="square",ec='white',fc='white',alpha=0.1))
    ax3=sns.heatmap(mean_cm1, annot=True, cmap="Blues", fmt=".2f",
                xticklabels=[],
                yticklabels=["Level 0", "Level 1", "Level 2", "Level 3"])
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
    ax3.set_yticklabels(ax3.get_yticklabels(), rotation=45, ha='right')
    plt.ylabel("True Label", fontsize=14)

    ax4 = plt.subplot(gs[0, 0])
    plt.text(-1.6,1.06,"a",size=14,rotation=0,ha="left",va="top",bbox=dict(boxstyle="square",ec='white',fc='white',alpha=0.1))
    plt.bar(xlabels, accuracies1, color=cmap(0), label='Accuracy')
    mean_accuracy= np.mean(accuracies1)
    plt.axhline(y=mean_accuracy, color='black', linestyle='--', label='Mean Accuracy')
    plt.ylabel('Accuracy', fontsize=14)
    plt.xlim(0.5, 23.5)
    plt.ylim(0.6, 1.005)
    plt.xticks(xlabels)
    legend = plt.legend(loc='lower right')
    legend.get_frame().set_alpha(0.95)
    plt.text(0.3, mean_accuracy, f'{mean_accuracy:.2f}', ha='right', va='center', color='black', fontsize=12)
    plt.tight_layout()
    plt.savefig('4class.png', dpi=500,pad_inches=0)
    plt.savefig('4class.pdf', pad_inches=0)

plt.show()