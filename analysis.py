# -*- coding: utf-8 -*-
"""
结果分析脚本：读取 train_all_models.py 输出的 results/metrics.json，
生成论文所需的对比曲线（图 1、图 2）与基线 / 改进模型的混淆矩阵（图 3）。

运行方式（在本脚本所在目录）：
    python analysis.py

前置条件：已先运行 python train_all_models.py 生成权重与 metrics.json。

验收要点（脚本会自动打印并写入 results/verification.txt）：
    1. 图 1、图 2 的曲线均含 30 个数据点（非空白图）；
    2. 混淆矩阵对角线合计 = 该 checkpoint 的测试准确率；
    3. 权重复算准确率 = 表中数字。
"""
import os
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from models import LeNet5, SEBNLenet

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
WEIGHT_DIR = os.path.join(BASE_DIR, 'weights')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 论文主对比：基线 vs 最终改进模型
COMPARE_PAIR = ['LeNet-5', 'SE-BN-LeNet']
COLORS = {'LeNet-5': '#1f77b4', 'SE-BN-LeNet': '#d62728'}


def load_metrics():
    path = os.path.join(RESULTS_DIR, 'metrics.json')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'未找到 {path}，请先运行 python train_all_models.py 完成训练。'
        )
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_comparison(metrics, verify):
    """绘制论文图 1（准确率对比）与图 2（损失对比），均为主种子 30 点曲线。"""
    models = metrics['models']
    for metric_key, ylabel, title, fname in [
        ('test_acc', 'Accuracy (%)', 'Accuracy Curves Comparison', 'accuracy_comparison.png'),
        ('test_loss', 'Loss', 'Loss Curves Comparison', 'loss_comparison.png'),
    ]:
        plt.figure(figsize=(9, 5), dpi=200)
        for name in COMPARE_PAIR:
            history = models[name]['history']
            epochs = range(1, len(history[metric_key]) + 1)
            npts = len(history[metric_key])
            verify[f'{fname}:{name}:points'] = npts
            plt.plot(epochs, history[metric_key], '-', color=COLORS[name], linewidth=2,
                     label=f'{name} (Test)')
            plt.plot(epochs, history['val_' + metric_key.split('_')[1]], '--',
                     color=COLORS[name], alpha=0.45, label=f'{name} (Val)')
        # 最终统计标注（单种子 / 多种子自适应）
        n_seeds = len(metrics['config']['seeds'])
        label = 'Single-run final' if n_seeds == 1 else f'{n_seeds}-seed final'
        txt = ' | '.join(
            (f"{name} {models[name]['test_acc_mean']:.2f}%" if n_seeds == 1 else
             f"{name} {models[name]['test_acc_mean']:.2f}±{models[name]['test_acc_std']:.2f}%")
            if metric_key == 'test_acc' else
            f"{name} {models[name]['test_loss_mean']:.4f}"
            for name in COMPARE_PAIR)
        plt.figtext(0.5, -0.02, f'{label}: {txt}', ha='center', fontsize=9)
        plt.xlabel('Epoch'); plt.ylabel(ylabel); plt.title(title)
        plt.legend(loc='best'); plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, fname), bbox_inches='tight')
        plt.close()
        print(f'已保存: results/{fname}')


def build_test_loader():
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_dataset = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)
    return DataLoader(test_dataset, batch_size=256, shuffle=False)


def confusion_matrix(model, loader):
    model.eval()
    cm = np.zeros((10, 10), dtype=int)
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, pred = torch.max(output, 1)
            for t, p in zip(target.cpu().numpy(), pred.cpu().numpy()):
                cm[t, p] += 1
    return cm


def plot_confusion(cm, name):
    plt.figure(figsize=(8, 7), dpi=200)
    plt.imshow(cm, cmap='Blues')
    for i in range(10):
        for j in range(10):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center',
                     color='white' if cm[i, j] > cm.max() * 0.5 else 'black', fontsize=8)
    plt.colorbar()
    plt.xlabel('Predicted Label'); plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {name}')
    plt.xticks(range(10)); plt.yticks(range(10)); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'confusion_{name}.png'))
    plt.close()
    print(f'已保存: results/confusion_{name}.png')

    per_class = cm.diagonal() / cm.sum(axis=1) * 100
    acc = cm.diagonal().sum() / cm.sum() * 100
    print(f'{name} 每类准确率: ' + ' '.join(f'{i}:{per_class[i]:.2f}' for i in range(10)))
    print(f'{name} 测试准确率（混淆矩阵对角线）: {acc:.2f}%')
    return acc, per_class


def main():
    metrics = load_metrics()
    print('读取到模型:', list(metrics['models'].keys()))
    verify = {}
    plot_comparison(metrics, verify)

    test_loader = build_test_loader()
    model_factory = {'LeNet-5': LeNet5, 'SE-BN-LeNet': SEBNLenet}
    for name, model_cls in model_factory.items():
        weight_path = os.path.join(WEIGHT_DIR, f'{name}_best.pth')
        if not os.path.exists(weight_path):
            print(f'跳过 {name}: 权重不存在')
            continue
        model = model_cls().to(device)
        model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
        cm = confusion_matrix(model, test_loader)
        acc, per_class = plot_confusion(cm, name)
        # 验收：对角线合计 vs metrics 中主种子结果
        primary = str(metrics['models'][name]['primary_seed'])
        reported = metrics['models'][name]['runs'][primary]['test_acc']
        verify[f'{name}:cm_acc'] = round(acc, 2)
        verify[f'{name}:reported_acc'] = reported
        verify[f'{name}:match'] = bool(abs(acc - reported) < 0.05)
        verify[f'{name}:per_class'] = [round(float(x), 2) for x in per_class]

    lines = ['=== 自动验收结果 ===']
    for k, v in verify.items():
        lines.append(f'{k}: {v}')
    report = '\n'.join(lines)
    with open(os.path.join(RESULTS_DIR, 'verification.txt'), 'w', encoding='utf-8') as f:
        f.write(report)
    print('\n' + report)


if __name__ == '__main__':
    main()
