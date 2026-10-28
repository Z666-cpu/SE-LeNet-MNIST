# -*- coding: utf-8 -*-
"""
结果分析脚本：读取 train_all_models.py 输出的 results/metrics.json，
生成论文所需的合并对比曲线，并基于保存的权重绘制混淆矩阵。

运行方式（在本脚本所在目录）：
    python analysis.py

前置条件：已先运行 python train_all_models.py 生成权重与 metrics.json。
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


def plot_comparison(metrics):
    """绘制论文图1（准确率对比）与图2（损失对比）。"""
    for metric_key, ylabel, title, fname in [
        ('test_acc', 'Accuracy (%)', 'Accuracy Curves Comparison', 'accuracy_comparison.png'),
        ('test_loss', 'Loss', 'Loss Curves Comparison', 'loss_comparison.png'),
    ]:
        plt.figure(figsize=(9, 5), dpi=200)
        for name in COMPARE_PAIR:
            history = metrics[name]['history']
            epochs = range(1, len(history[metric_key]) + 1)
            plt.plot(epochs, history[metric_key], '-', color=COLORS[name], label=f'{name} (Test)')
            train_key = metric_key.replace('test', 'train')
            plt.plot(epochs, history[train_key], '--', color=COLORS[name], alpha=0.5,
                     label=f'{name} (Train)')
        plt.xlabel('Epoch'); plt.ylabel(ylabel); plt.title(title)
        plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, fname))
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
                     color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=9)
    plt.colorbar()
    plt.xlabel('Predicted Label'); plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {name}')
    plt.xticks(range(10)); plt.yticks(range(10)); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'confusion_{name}.png'))
    plt.close()
    print(f'已保存: results/confusion_{name}.png')

    per_class = cm.diagonal() / cm.sum(axis=1) * 100
    print(f'{name} 每类准确率: ' + ', '.join(f'{i}:{per_class[i]:.2f}%' for i in range(10)))
    acc = cm.diagonal().sum() / cm.sum() * 100
    print(f'{name} 测试准确率: {acc:.2f}%')


def main():
    metrics = load_metrics()
    print('读取到模型指标:', list(metrics.keys()))
    plot_comparison(metrics)

    test_loader = build_test_loader()
    model_factory = {
        'LeNet-5': LeNet5,
        'SE-BN-LeNet': SEBNLenet,
    }
    for name, model_cls in model_factory.items():
        weight_path = os.path.join(WEIGHT_DIR, f'{name}_best.pth')
        if not os.path.exists(weight_path):
            print(f'跳过 {name}: 权重 {weight_path} 不存在')
            continue
        model = model_cls().to(device)
        model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
        cm = confusion_matrix(model, test_loader)
        plot_confusion(cm, name)


if __name__ == '__main__':
    main()
