"""
# @file name  : mobilenet train-valid-test.py
# @author     : Ziyao Ling
# @date       : 01/12/2024
# @brief      : mobilenet ttrain-valid-test
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score, top_k_accuracy_score
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import time
import pickle
from torch.cuda.amp import autocast, GradScaler
from collections import Counter
from model_thesis.mobilenet import MobileNetMultiTask
from datasetRes import PorcelainMultiTaskDataset, get_transforms


# ================== Loss Function ==================
class MultiTaskLoss(nn.Module):
    def __init__(self, class_weights_dict, task_weights=None):
        super(MultiTaskLoss, self).__init__()

        # Task Weight
        self.task_weights = task_weights or {
            'dynasty': 1.0,
            'kiln': 1.0,
            'glaze': 1.0,
            'type': 1.0
        }

        # crossentropyloss for each task
        self.criteria = {}
        for task, weights in class_weights_dict.items():
            if weights is not None and torch.cuda.is_available():
                weights = weights.cuda()
            self.criteria[task] = nn.CrossEntropyLoss(weight=weights)

    def forward(self, outputs, targets):
        losses = {}
        for task in self.criteria:
            losses[task] = self.criteria[task](outputs[task], targets[task])

        # total_loss
        total_loss = sum(self.task_weights[task] * losses[task] for task in losses)
        losses['total'] = total_loss

        return losses


# ================== train function ==================
def train_epoch(model, loader, criterion, optimizer, device, scaler=None, use_amp=False):
    model.train()
    running_losses = {task: 0.0 for task in ['total', 'dynasty', 'kiln', 'glaze', 'type']}
    correct_preds = {task: 0 for task in ['dynasty', 'kiln', 'glaze', 'type']}
    total_samples = 0

    # class distribution monitor
    class_counters = {task: Counter() for task in ['dynasty', 'kiln', 'glaze', 'type']}

    pbar = tqdm(loader, desc='Training')
    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}

        # class_counter
        for task in class_counters:
            class_counters[task].update(targets[task].cpu().numpy())

        optimizer.zero_grad()

        # amp
        if use_amp and scaler is not None:
            with autocast():
                outputs = model(images)
                losses = criterion(outputs, targets)

            scaler.scale(losses['total']).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            losses = criterion(outputs, targets)
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # statics
        batch_size = images.size(0)
        total_samples += batch_size

        for task in ['dynasty', 'kiln', 'glaze', 'type']:
            running_losses[task] += losses[task].item() * batch_size
            _, preds = torch.max(outputs[task], 1)
            correct_preds[task] += (preds == targets[task]).sum().item()
        running_losses['total'] += losses['total'].item() * batch_size

        # update
        if batch_idx % 10 == 0:
            current_loss = running_losses['total'] / total_samples
            current_acc = sum(correct_preds.values()) / (total_samples * 4)  # 平均准确率
            pbar.set_postfix({'loss': f'{current_loss:.4f}', 'acc': f'{current_acc:.3f}'})

    # avg_loss
    avg_losses = {k: v / total_samples for k, v in running_losses.items()}
    accuracies = {k: v / total_samples for k, v in correct_preds.items()}

    return avg_losses, accuracies, class_counters


# ================== Valid function ==================
def validate(model, loader, criterion, device):
    model.eval()
    running_losses = {task: 0.0 for task in ['total', 'dynasty', 'kiln', 'glaze', 'type']}
    all_preds = {task: [] for task in ['dynasty', 'kiln', 'glaze', 'type']}
    all_targets = {task: [] for task in ['dynasty', 'kiln', 'glaze', 'type']}
    all_probs = {task: [] for task in ['dynasty', 'kiln', 'glaze', 'type']}

    with torch.no_grad():
        for images, targets in tqdm(loader, desc='Validating'):
            images = images.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}

            outputs = model(images)
            losses = criterion(outputs, targets)

            # total_loss
            batch_size = images.size(0)
            for task in running_losses:
                if task == 'total':
                    running_losses[task] += losses[task].item() * batch_size
                else:
                    running_losses[task] += losses[task].item() * batch_size

                    # results_saving
                    probs = torch.softmax(outputs[task], dim=1)
                    _, preds = torch.max(outputs[task], 1)

                    all_preds[task].extend(preds.cpu().numpy())
                    all_targets[task].extend(targets[task].cpu().numpy())
                    all_probs[task].extend(probs.cpu().numpy())

    # avg_loss
    total_samples = len(loader.dataset)
    avg_losses = {k: v / total_samples for k, v in running_losses.items()}

    # numpy
    for task in all_preds:
        all_preds[task] = np.array(all_preds[task])
        all_targets[task] = np.array(all_targets[task])
        all_probs[task] = np.array(all_probs[task])

    return avg_losses, all_preds, all_targets, all_probs


# ================== evaluation_function ==================
def evaluate_predictions(all_targets, all_preds, all_probs, num_classes_dict):
    results = {}

    for task in all_targets:
        y_true = all_targets[task]
        y_pred = all_preds[task]
        y_prob = all_probs[task]

        # Top-1 acc
        acc_top1 = accuracy_score(y_true, y_pred)

        # Top-5 acc
        if num_classes_dict[task] >= 5:
            acc_top5 = top_k_accuracy_score(y_true, y_prob, k=5)
        else:
            acc_top5 = acc_top1

        # F1
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # Pre & recall
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)

        # Con
        cm = confusion_matrix(y_true, y_pred)

        results[task] = {
            'accuracy': acc_top1,
            'top5_accuracy': acc_top5,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': cm
        }

    return results


# ================== Other function ==================
def create_label_encoders(train_df, label_cols):
    
    label_encoders = {}
    for col in label_cols:
        le = LabelEncoder()
        le.fit(train_df[col])
        label_encoders[col] = le
    return label_encoders


def compute_class_weights_improved(train_df, label_cols, label_encoders, strategy='effective'):
    """

    Strategies:
        - 'balanced'
        - 'effective': Effective Number of Samples 
        - 'sqrt'
    """
    class_weights = {}

    for col in label_cols:
        encoded_labels = label_encoders[col].transform(train_df[col])
        unique_classes, class_counts = np.unique(encoded_labels, return_counts=True)

        if strategy == 'balanced':
            # sklearn
            weights = len(encoded_labels) / (len(unique_classes) * class_counts)

        elif strategy == 'effective':
            # Effective Number of Samples
            beta = 0.999
            effective_num = (1 - beta ** class_counts) / (1 - beta)
            weights = 1.0 / effective_num
            weights = weights / weights.sum() * len(unique_classes)

        elif strategy == 'sqrt':
            # sqrt
            weights = 1.0 / np.sqrt(class_counts)
            weights = weights / weights.mean()

        # constrain max weight
        max_weight = 10.0
        weights = np.minimum(weights, max_weight)

        class_weights[col] = torch.tensor(weights, dtype=torch.float)

        # print
        print(f"\n{col} class weights ({strategy}):")
        print(f"  Min weight: {weights.min():.3f}")
        print(f"  Max weight: {weights.max():.3f}")
        print(f"  Weight ratio: {weights.max() / weights.min():.1f}x")

    return class_weights


def create_balanced_sampler(train_df, label_cols, strategy='most_imbalanced'):
    """

    Strategies:
        - 'dynasty'
        - 'glaze'
        - 'combined'
        - 'most_imbalanced'
    """
    from collections import Counter

    print(f"\n⚖️ Creating balanced sampler (strategy: {strategy})...")

    if strategy == 'combined':
        # label
        combined_labels = train_df[label_cols].apply(
            lambda x: '_'.join(x.astype(str)), axis=1
        )
        label_counts = Counter(combined_labels)

        # weight
        weights = 1.0 / np.array([label_counts[label] for label in combined_labels])
        print(f"Combined classes: {len(label_counts)}")

    elif strategy == 'most_imbalanced':
        # most imbalanced
        imbalance_ratios = {}
        for col in label_cols:
            counts = train_df[col].value_counts()
            imbalance_ratios[col] = counts.max() / counts.min()

        most_imbalanced_task = max(imbalance_ratios, key=imbalance_ratios.get)
        print(f"Most imbalanced task: {most_imbalanced_task} (ratio: {imbalance_ratios[most_imbalanced_task]:.2f})")
        for task, ratio in imbalance_ratios.items():
            print(f"  {task}: {ratio:.2f}x")

        label_counts = train_df[most_imbalanced_task].value_counts().to_dict()
        weights = 1.0 / train_df[most_imbalanced_task].map(label_counts)

    else:
        # single task
        label_counts = train_df[strategy].value_counts().to_dict()
        weights = 1.0 / train_df[strategy].map(label_counts)
        print(f"Sampling based on {strategy}: {len(label_counts)} classes")

    # normalization
    weights = weights / weights.sum() * len(weights)

    # check weight type
    if hasattr(weights, 'values'):  # pandas Series
        weights_array = weights.values
    elif isinstance(weights, np.ndarray):  # numpy array
        weights_array = weights
    else:  
        weights_array = np.array(weights)

    sampler = WeightedRandomSampler(
        weights=weights_array,
        num_samples=len(weights_array),
        replacement=True
    )

    print(f"Sampler created with {len(weights)} samples")

    return sampler


def save_results(results, save_path):
    """save Json results"""
    # numpy
    processed_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            processed_results[key] = {}
            for k, v in value.items():
                if isinstance(v, np.ndarray):
                    processed_results[key][k] = v.tolist()
                else:
                    processed_results[key][k] = v
        else:
            processed_results[key] = value

    # save as json
    with open(save_path, 'w') as f:
        json.dump(processed_results, f, indent=2)


# label translation dictionary
LABEL_TRANSLATIONS = {
    # Dynasty (朝代)
    '宋代': 'Song',
    '元代': 'Yuan',

    # Kiln (窑口) - 主要窑口
    '景德镇窑': 'Jing',
    '龙泉窑': 'Long',
    '汝窑': 'Ru',
    '钧窑': 'Jun',
    '定窑': 'Ding',
    '哥窑': 'Ge',
    '官窑': 'Guan',
    '磁州窑': 'CiZ',
    '耀州窑': 'YaoZ',
    '建窑': 'Jian',
    '广窑': 'Guang',
    '霍州窑': 'Huo',
    '临川窑': 'LinC',
    '德化窑': 'DeH',
    '湘湖窑': 'Xiang',
    '彭窑': 'Peng',
    '吉州窑': 'JiZ',

    # Glaze (釉色)
    '青釉': 'Celadon',
    '白釉': 'White',
    '黑釉': 'Black',
    '青白釉': 'BluishW',
    '青花': 'BlueW',
    '黄绿釉': 'GreenY',
    '灰青釉': 'GreyC',
    '酱釉': 'Brown',
    '绿釉': 'Green',
    '玫瑰紫釉': 'RoseP',
    '葡萄紫釉': 'GrapeP',
    '青黄釉': 'YellowG',
    '天蓝釉': 'SkyB',
    '天青釉': 'BlueG',
    '透明釉': 'Tran',
    '月白釉': 'MoonW',
    '牙白釉': 'IvoryW',

    # Type (器型)
    '碗': 'Bowl',
    '盘': 'Plate',
    '瓶': 'Vase',
    '罐': 'Jar',
    '壶': 'Pot',
    '杯': 'Cup',
    '盏': 'TeaB',
    '盏托': 'TeaB-S',
    '洗': 'Washer',
    '炉': 'Censer',
    '笔筒': 'Brush',
    '尊': 'Zun',
    '水盂': 'Basin',
    '枕': 'Pillow',
    '盆': 'Plan',
    '盆托': 'PlanS',
    '水丞': 'Cheng',
    '簋': 'Gui',
    '盒': 'Box',
    '碟': 'Dish',
}


def translate_labels(label_encoder, task_name):
    """translate chinese"""
    original_labels = label_encoder.classes_
    translated_labels = []

    for i, label in enumerate(original_labels):
        # label_clean
        label_clean = str(label).strip()

        if label_clean in LABEL_TRANSLATIONS:
            translated = LABEL_TRANSLATIONS[label_clean]
        else:
            task_prefix = {
                'dynasty': 'D',
                'kiln': 'K',
                'glaze': 'G',
                'type': 'T'
            }
            translated = f"{task_prefix.get(task_name, 'C')}{i}"
            print(f"Warning: No translation for '{label}', using '{translated}'")

        translated_labels.append(translated)

    return translated_labels


def create_label_mapping_table(chinese_labels, english_labels, task_name, save_dir):
    """create_label_mapping_label"""
    import pandas as pd

    mapping_df = pd.DataFrame({
        'Index': range(len(chinese_labels)),
        'Chinese': chinese_labels,
        'English': english_labels
    })

    # CSV
    mapping_df.to_csv(os.path.join(save_dir, f'{task_name}_label_mapping.csv'),
                      index=False, encoding='utf-8-sig')

    # visualize
    if len(chinese_labels) <= 25:
        fig, ax = plt.subplots(figsize=(8, min(len(chinese_labels) * 0.5 + 2, 14)))
        ax.axis('tight')
        ax.axis('off')

        table_data = mapping_df.values.tolist()
        table = ax.table(cellText=table_data,
                         colLabels=mapping_df.columns,
                         cellLoc='center',
                         loc='center')

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        # format
        for i in range(len(mapping_df.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        plt.title(f'{task_name.capitalize()} Label Mapping', fontsize=14, pad=20)
        plt.savefig(os.path.join(save_dir, f'{task_name}_label_mapping.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()


def plot_confusion_matrix_detailed(cm, label_encoder, task_name, save_dir, dataset_type='test'):
    """con"""
    # en_label
    english_labels = translate_labels(label_encoder, task_name)
    n_classes = len(english_labels)

    # adjust size
    if n_classes <= 5:
        fig_size = (10, 8)
    elif n_classes <= 10:
        fig_size = (12, 10)
    elif n_classes <= 20:
        fig_size = (16, 14)
    else:
        fig_size = (20, 18)

    # 1. conf
    plt.figure(figsize=fig_size)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=english_labels,
                yticklabels=english_labels,
                cbar_kws={'label': 'Count'})

    plt.title(f'{task_name.capitalize()} - {dataset_type.capitalize()} Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)

    # rotate
    if n_classes > 10:
        plt.xticks(rotation=45, ha='right')
    else:
        plt.xticks(rotation=30, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()

    # save
    filename = f'{dataset_type}_{task_name}_confusion_matrix.png'
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. normalization_matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  

    plt.figure(figsize=fig_size)
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=english_labels,
                yticklabels=english_labels,
                cbar_kws={'label': 'Percentage'},
                vmin=0, vmax=1)

    plt.title(f'{task_name.capitalize()} - {dataset_type.capitalize()} Normalized Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)

    if n_classes > 10:
        plt.xticks(rotation=45, ha='right')
    else:
        plt.xticks(rotation=30, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()

    # save
    filename_norm = f'{dataset_type}_{task_name}_confusion_matrix_normalized.png'
    plt.savefig(os.path.join(save_dir, filename_norm), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. create
    chinese_labels = label_encoder.classes_
    create_label_mapping_table(chinese_labels, english_labels, task_name, save_dir)
    print(f"\n  Confusion Matrix Analysis for {task_name}:")

    # conf_paris
    confusion_pairs = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append((cm[i, j], i, j))

    confusion_pairs.sort(reverse=True)

    if confusion_pairs:
        print(f"  Top confusions:")
        for count, i, j in confusion_pairs[:5]:  # top5
            true_label = english_labels[i]
            pred_label = english_labels[j]
            percentage = (count / cm[i].sum() * 100) if cm[i].sum() > 0 else 0
            print(f"    {true_label} → {pred_label}: {count} times ({percentage:.1f}%)")

    return cm_normalized


def save_results(results, save_path):
    """save"""
    # process numpy
    processed_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            processed_results[key] = {}
            for k, v in value.items():
                if isinstance(v, np.ndarray):
                    processed_results[key][k] = v.tolist()
                else:
                    processed_results[key][k] = v
        else:
            processed_results[key] = value

    with open(save_path, 'w') as f:
        json.dump(processed_results, f, indent=2)

def monitor_class_distribution(class_counters, label_encoders, epoch, save_dir=None):
    """
    monitor 
    """
    # font
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    
    try:
        # Windows系统
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    except:
        
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

    plt.rcParams['axes.unicode_minus'] = False  

    print(f"\n📊 Epoch {epoch} - Class Distribution Analysis:")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    distribution_stats = {}

    for idx, (task, counter) in enumerate(class_counters.items()):
        ax = axes[idx]

        # sum
        all_classes = list(range(len(label_encoders[task].classes_)))
        counts = [counter.get(c, 0) for c in all_classes]
        total = sum(counts)

        # statistic
        if total > 0:
            percentages = [c / total * 100 for c in counts]
            min_pct = min(p for p in percentages if p > 0)
            max_pct = max(percentages)
            ratio = max_pct / min_pct if min_pct > 0 else float('inf')

            # entropy
            entropy = -sum(p / 100 * np.log(p / 100) if p > 0 else 0 for p in percentages)
            max_entropy = np.log(len(all_classes))
            uniformity = entropy / max_entropy if max_entropy > 0 else 0

            distribution_stats[task] = {
                'min_pct': min_pct,
                'max_pct': max_pct,
                'ratio': ratio,
                'uniformity': uniformity,
                'zero_count_classes': sum(1 for c in counts if c == 0)
            }

            # print
            print(f"\n{task.upper()}:")
            print(f"  Total samples: {total}")
            print(f"  Min representation: {min_pct:.2f}%")
            print(f"  Max representation: {max_pct:.2f}%")
            print(f"  Max/Min ratio: {ratio:.2f}x")
            print(f"  Distribution uniformity: {uniformity:.3f}")
            print(f"  Classes not seen: {distribution_stats[task]['zero_count_classes']}")

            
            labels = []
            for c in all_classes:
                try:
                    label = label_encoders[task].inverse_transform([c])[0]
                    
                    if any(ord(ch) > 127 for ch in str(label)):
                        label = f"Class_{c}"
                except:
                    label = f"Class_{c}"
                labels.append(label)

            # bar graph
            sorted_indices = np.argsort(counts)[::-1]
            sorted_counts = [counts[i] for i in sorted_indices]
            sorted_labels = [labels[i] for i in sorted_indices]
            sorted_percentages = [percentages[i] for i in sorted_indices]

            bars = ax.bar(range(len(sorted_counts)), sorted_counts)

            colors = ['red' if p < 1.0 else 'orange' if p < 5.0 else 'skyblue'
                      for p in sorted_percentages]
            for bar, color in zip(bars, colors):
                bar.set_color(color)

            ax.set_xlabel('Classes (sorted by frequency)')
            ax.set_ylabel('Sample Count')
         
            task_name_en = {
                'dynasty': 'Dynasty',
                'kiln': 'Kiln',
                'glaze': 'Glaze',
                'type': 'Type'
            }
            ax.set_title(f'{task_name_en.get(task, task)} - Distribution (Uniformity: {uniformity:.3f})')

        
            if len(sorted_labels) > 20:
                step = max(1, len(sorted_labels) // 20)
                ax.set_xticks(range(0, len(sorted_labels), step))
                ax.set_xticklabels([sorted_labels[i] for i in range(0, len(sorted_labels), step)],
                                   rotation=45, ha='right')
            else:
                ax.set_xticks(range(len(sorted_labels)))
                ax.set_xticklabels(sorted_labels, rotation=45, ha='right')

            ideal_count = total / len(all_classes)
            ax.axhline(y=ideal_count, color='green', linestyle='--', alpha=0.5,
                       label=f'Ideal uniform: {ideal_count:.1f}')
            ax.legend()

        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Class Distribution at Epoch {epoch}', fontsize=16)
    plt.tight_layout()

    if save_dir:
        plt.savefig(os.path.join(save_dir, f'class_distribution_epoch_{epoch}.png'),
                    dpi=150, bbox_inches='tight')

    plt.close()

    return distribution_stats


def analyze_sampling_effectiveness(distribution_stats_history, save_dir=None):
    """
    analyze_sampling_effectiveness
    """
    epochs = list(range(1, len(distribution_stats_history) + 1))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, task in enumerate(['dynasty', 'kiln', 'glaze', 'type']):
        ax = axes[idx]

        # epoch
        uniformities = [stats[task]['uniformity'] for stats in distribution_stats_history]
        ratios = [stats[task]['ratio'] for stats in distribution_stats_history]

        # twin y
        ax2 = ax.twinx()

        line1 = ax.plot(epochs, uniformities, 'b-', marker='o', label='Uniformity')
        ax.set_ylabel('Distribution Uniformity', color='b')
        ax.tick_params(axis='y', labelcolor='b')
        ax.set_ylim(0, 1)

     
        line2 = ax2.plot(epochs, ratios, 'r--', marker='s', label='Max/Min Ratio')
        ax2.set_ylabel('Max/Min Class Ratio', color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        ax.set_xlabel('Epoch')
        task_name_en = {
            'dynasty': 'Dynasty',
            'kiln': 'Kiln',
            'glaze': 'Glaze',
            'type': 'Type'
        }
        ax.set_title(f'{task_name_en.get(task, task)} - Sampling Effectiveness')
        ax.grid(True, alpha=0.3)

        # layout
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='center right')

    plt.suptitle('Sampling Strategy Effectiveness Over Time', fontsize=14)
    plt.tight_layout()

    if save_dir:
        plt.savefig(os.path.join(save_dir, 'sampling_effectiveness.png'),
                    dpi=150, bbox_inches='tight')

    plt.close()


def plot_training_curves(history, save_dir):
    """training curve"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))  
    axes = axes.flatten()  

    # Loss curve
    ax = axes[0]
    ax.plot(history['train_loss'], label='Train Loss')
    ax.plot(history['val_loss'], label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Total Loss')
    ax.legend()
    ax.grid(True)

    # acc per task
    tasks = ['dynasty', 'kiln', 'glaze', 'type']
    for idx, task in enumerate(tasks):
        ax = axes[idx + 1]  

        train_key = f'train_acc_{task}'
        val_key = f'val_acc_{task}'

        if train_key in history:
            ax.plot(history[train_key], label='Train')
        if val_key in history:
            ax.plot(history[val_key], label='Val')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{task.capitalize()} Accuracy')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300)
    plt.close()


# ================== training function ==================
def train_mobilenet_multitask(config):
    """train"""

    print("=" * 60)
    print("ResNet Multi-Task Training")
    print("=" * 60)
    print(f"Model: {config['model_name']}")
    print(f"Pretrained: {config['pretrained']}")
    print(f"Device: {config['device']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Epochs: {config['num_epochs']}")
    print(f"Mixed Precision: {config['use_amp']}")
    print("=" * 60)

    # save_dir
    save_dir = Path(config['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)

    # load data
    print("\n📂 Loading data...")
    train_df = pd.read_csv(config['train_csv'])
    val_df = pd.read_csv(config['val_csv'])
    test_df = pd.read_csv(config['test_csv'])

    # clean label
    label_cols = config['label_cols']
    for col in label_cols:
        train_df[col] = train_df[col].astype(str).str.strip()
        val_df[col] = val_df[col].astype(str).str.strip()
        test_df[col] = test_df[col].astype(str).str.strip()

    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")

    # encoder 
    print("\n🏷️ Creating label encoders...")
    label_encoders = create_label_encoders(train_df, label_cols)
    num_classes = {task: len(le.classes_) for task, le in label_encoders.items()}

    print("Number of classes:")
    for task, n in num_classes.items():
        print(f"  {task}: {n}")

    # class_weight
    class_weights = compute_class_weights_improved(
        train_df, label_cols, label_encoders,
        strategy='effective'  # effective number
    )

    # save label encoder
    with open(save_dir / 'label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    print("✅ Label encoders saved")

    # class_balanced_sampler
    sampler = create_balanced_sampler(
        train_df, label_cols,
        strategy='glaze'
    )

    # dataset
    print("\n📊 Creating datasets...")
    train_dataset = PorcelainMultiTaskDataset(
        train_df, config['train_img_dir'], label_encoders,
        transform=get_transforms(train=True)
    )
    val_dataset = PorcelainMultiTaskDataset(
        val_df, config['val_img_dir'], label_encoders,
        transform=get_transforms(train=False)
    )
    test_dataset = PorcelainMultiTaskDataset(
        test_df, config['test_img_dir'], label_encoders,
        transform=get_transforms(train=False)
    )

    # dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=sampler,  # sampler
        num_workers=config['num_workers'],
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    # create model 
    print(f"\n🏗️ Creating model...")
    model = MobileNetMultiTask(
        num_classes_dict={'dynasty': 2, 'kiln': 17, 'glaze': 17, 'type': 20},
        model_name=config['model_name'],  # read config
        pretrained=config['pretrained']
    ).to(config['device'])

    # model parameter
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")



    # optimizer
    if config['pretrained']:
        # different learning rate
        backbone_params = []
        other_params = []

        for name, param in model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                other_params.append(param)

        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': config['lr_backbone']},
            {'params': other_params, 'lr': config['lr_head']}
        ], weight_decay=config['weight_decay'])
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['lr_head'],
            weight_decay=config['weight_decay']
        )

    # lr_scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs'],
        eta_min=1e-6
    )

    # loss_function
    criterion = MultiTaskLoss(class_weights, task_weights=config.get('task_weights'))

    if config['use_amp']:
        scaler = GradScaler('cuda')  # 新API
    else:
        scaler = None

    # traim_loop
    print("\n🚀 Starting training...")
    best_val_f1 = 0
    best_epoch = 0
    patience_counter = 0
    start_time = time.time()
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc_avg': [],
        'val_acc_avg': []
    }
    # history
    for task in label_cols:
        history[f'train_acc_{task}'] = []
        history[f'val_acc_{task}'] = []

    # monitor
    distribution_stats_history = []

    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch}/{config['num_epochs']} - LR: {scheduler.get_last_lr()[0]:.6f}")

        train_losses, train_accs, class_counters = train_epoch(
            model, train_loader, criterion, optimizer,
            config['device'], scaler, config['use_amp']
        )

    
        if epoch == 1 or epoch % 5 == 0:
            distribution_stats = monitor_class_distribution(
                class_counters, label_encoders, epoch, save_dir
            )
            distribution_stats_history.append(distribution_stats)

        val_losses, val_preds, val_targets, val_probs = validate(
            model, val_loader, criterion, config['device']
        )

        val_results = evaluate_predictions(val_targets, val_preds, val_probs, num_classes)

        # avg_metric
        avg_train_acc = np.mean(list(train_accs.values()))
        avg_val_acc = np.mean([res['accuracy'] for res in val_results.values()])
        avg_val_f1 = np.mean([res['f1_macro'] for res in val_results.values()])

        # record history
        history['train_loss'].append(train_losses['total'])
        history['val_loss'].append(val_losses['total'])
        history['train_acc_avg'].append(avg_train_acc)
        history['val_acc_avg'].append(avg_val_acc)

        for task in label_cols:
            history[f'train_acc_{task}'].append(train_accs[task])
            history[f'val_acc_{task}'].append(val_results[task]['accuracy'])

        # print
        print(f"\nTrain Loss: {train_losses['total']:.4f}, Avg Acc: {avg_train_acc:.3f}")
        print(f"Val Loss: {val_losses['total']:.4f}, Avg Acc: {avg_val_acc:.3f}")
        print(f"\nPer-task validation results:")
        for task in label_cols:
            print(f"  {task}: Acc={val_results[task]['accuracy']:.3f}, "
                  f"F1={val_results[task]['f1_macro']:.3f}")

        # save best model
        if avg_val_f1 > best_val_f1:
            best_val_f1 = avg_val_f1
            best_epoch = epoch
            patience_counter = 0

            print(f"\n💾 Saving best model (F1={best_val_f1:.3f})...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_f1': best_val_f1,
                'val_results': val_results,
                'config': config
            }, save_dir / 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"\n⏹️ Early stopping triggered!")
                break

        # scheduler_step
        scheduler.step()

        # checkpoint
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }, save_dir / f'checkpoint_epoch_{epoch}.pth')
    training_time = (time.time() - start_time) / 60
    print(f"\n✅ Training completed in {training_time:.1f} minutes")
    print(f"Best epoch: {best_epoch} with avg F1: {best_val_f1:.3f}")

    # training curve
    plot_training_curves(history, save_dir)

    # analyze_sampling_effectivenes
    if distribution_stats_history:
        analyze_sampling_effectiveness(distribution_stats_history, save_dir)

    # test_best_model
    print("\n📊 Testing best model...")
    checkpoint = torch.load(save_dir / 'best_model.pth', map_location=config['device'], weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_losses, test_preds, test_targets, test_probs = validate(
        model, test_loader, criterion, config['device']
    )
    test_results = evaluate_predictions(test_targets, test_preds, test_probs, num_classes)

    # print
    print("\n🎯 Test Results:")
    test_summary = {}

    # label_encoders for conf
    with open(save_dir / 'label_encoders.pkl', 'rb') as f:
        saved_label_encoders = pickle.load(f)

    for task in label_cols:
        print(f"\n{task.upper()}:")
        print(f"  Accuracy: {test_results[task]['accuracy']:.3f}")
        print(f"  Top-5 Acc: {test_results[task]['top5_accuracy']:.3f}")
        print(f"  F1 (macro): {test_results[task]['f1_macro']:.3f}")
        print(f"  F1 (weighted): {test_results[task]['f1_weighted']:.3f}")
        print(f"  Precision: {test_results[task]['precision']:.3f}")
        print(f"  Recall: {test_results[task]['recall']:.3f}")

        test_summary[task] = {
            'accuracy': test_results[task]['accuracy'],
            'top5_accuracy': test_results[task]['top5_accuracy'],
            'f1_macro': test_results[task]['f1_macro'],
            'f1_weighted': test_results[task]['f1_weighted'],
            'precision': test_results[task]['precision'],
            'recall': test_results[task]['recall']
        }

        # conf
        cm = test_results[task]['confusion_matrix']
        plot_confusion_matrix_detailed(
            cm,
            saved_label_encoders[task],
            task,
            save_dir,
            dataset_type='test'
        )
    # save
    final_results = {
        'config': config,
        'best_epoch': best_epoch,
        'best_val_f1': best_val_f1,
        'training_time_minutes': training_time,
        'test_results': test_summary,
        'history': history
    }

    save_results(final_results, save_dir / 'final_results.json')




# ================== config and running ==================
if __name__ == "__main__":
    # config
    config = {
        # data_dir
        'train_csv': r"",
        'val_csv': r"",
        'test_csv': r"",
        'train_img_dir': r"",
        'val_img_dir': r"",
        'test_img_dir': r"",
        'save_dir': r"",

        # model config
        "model": MobileNetV3Large
        'pretrained': True,

        # train config
        'batch_size': 64, 
        'num_epochs': 50,  
        'lr_backbone': 1e-4,  
        'lr_head': 1e-3,            
        'weight_decay': 1e-5,       
        'patience': 10,

        # task weight(static)
        'task_weights': {
            'dynasty': 1.0,  
            'kiln': 1.2,  
            'glaze': 2.0,
            'type': 1.5  
        },

        # class_weight_strategy
        'class_weight_strategy': 'effective',  # effective number
        'sampling_strategy': 'glaze',  # most difficult task

        # other
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers': 4,
        'use_amp': False,  
        'label_cols': ['dynasty', 'kiln', 'glaze', 'type']
    }

    # run
    test_results = train_mobilenet_multitask(config)
