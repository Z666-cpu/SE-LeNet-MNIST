# -*- coding: utf-8 -*-
"""
一键复现主脚本：训练论文中的全部 4 个模型（基线 + 3 个变体，含消融）。

模型：
  LeNet-5        基线模型
  SE-LeNet       仅 SE 通道注意力（消融）
  BN-LeNet       仅批归一化（消融）
  SE-BN-LeNet    批归一化 + 通道注意力（最终改进模型）

运行方式（在本脚本所在目录）：
    python train_all_models.py

输出：
    weights/<模型名>_best.pth      最优模型权重
    results/metrics.json           全部模型逐轮训练指标（供 analysis.py 绘图）
    results/<模型名>_accuracy.png  单模型准确率曲线
    results/<模型名>_loss.png      单模型损失曲线
    train_log.txt                  人类可读训练日志（与论文表格口径一致）
"""
import os
import time
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from models import LeNet5, SELenet, BNLenet, SEBNLenet

# ===================== 固定实验配置（与论文一致，请勿修改） =====================
EPOCHS = 30
BATCH_SIZE = 64
LR = 0.001
WEIGHT_DECAY = 1e-4
SEED = 42
NUM_WORKERS = 0          # Windows 下建议 0；Linux 可调到 2~4
# ==============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
WEIGHT_DIR = os.path.join(BASE_DIR, 'weights')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(WEIGHT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def setup_determinism(seed: int):
    """固定全部随机源，开启 cuDNN 确定性，保证跨运行可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def seed_worker(worker_id):
    """DataLoader 子进程随机种子初始化。"""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main():
    setup_determinism(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'当前使用设备: {device}')
    if torch.cuda.is_available():
        print(f'GPU名称: {torch.cuda.get_device_name(0)}')
    print('=' * 60)

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    print('正在加载 MNIST 数据集，首次运行会自动下载...')
    train_dataset = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)

    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, worker_init_fn=seed_worker, generator=g,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, worker_init_fn=seed_worker,
    )
    print(f'数据集加载完成！训练集 {len(train_dataset)} 张，测试集 {len(test_dataset)} 张')
    print('=' * 60)

    model_factory = {
        'LeNet-5': LeNet5,
        'SE-LeNet': SELenet,
        'BN-LeNet': BNLenet,
        'SE-BN-LeNet': SEBNLenet,
    }

    summary = {}
    log_lines = []

    for model_name, model_cls in model_factory.items():
        # 每个模型重新初始化权重，保证相互独立、种子可控
        setup_determinism(SEED)
        model = model_cls().to(device)
        params = sum(p.numel() for p in model.parameters())

        print(f'\n开始训练 {model_name} 模型，共 {EPOCHS} 轮，参数量 {params}')
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
        best_acc = 0.0
        best_test_loss = float('inf')
        t0 = time.time()

        for epoch in range(EPOCHS):
            # 训练阶段
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * data.size(0)
                _, pred = torch.max(output.data, 1)
                train_total += target.size(0)
                train_correct += (pred == target).sum().item()
            train_loss /= train_total
            train_acc = 100. * train_correct / train_total

            # 测试阶段
            model.eval()
            test_loss, test_correct, test_total = 0.0, 0, 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = criterion(output, target)
                    test_loss += loss.item() * data.size(0)
                    _, pred = torch.max(output.data, 1)
                    test_total += target.size(0)
                    test_correct += (pred == target).sum().item()
            test_loss /= test_total
            test_acc = 100. * test_correct / test_total

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)

            if test_acc > best_acc:
                best_acc = test_acc
                best_test_loss = test_loss
                torch.save(model.state_dict(), os.path.join(WEIGHT_DIR, f'{model_name}_best.pth'))

            line = (f'Epoch {epoch + 1:2d}/{EPOCHS} | 训练损失: {train_loss:.4f} | '
                    f'训练准确率: {train_acc:.2f}% | 测试损失: {test_loss:.4f} | '
                    f'测试准确率: {test_acc:.2f}%')
            print(line)
            log_lines.append(line)
            scheduler.step()

        elapsed = time.time() - t0
        print(f'{model_name} 训练完成！最优测试准确率: {best_acc:.2f}%，耗时 {elapsed:.1f} 秒')

        # 单模型训练曲线
        plt.figure(dpi=200)
        plt.plot(range(1, EPOCHS + 1), history['train_acc'], label='Train Accuracy')
        plt.plot(range(1, EPOCHS + 1), history['test_acc'], label='Test Accuracy')
        plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)')
        plt.title(f'{model_name} Accuracy Curve'); plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(RESULTS_DIR, f'{model_name}_accuracy.png'), bbox_inches='tight')
        plt.close()

        plt.figure(dpi=200)
        plt.plot(range(1, EPOCHS + 1), history['train_loss'], label='Train Loss')
        plt.plot(range(1, EPOCHS + 1), history['test_loss'], label='Test Loss')
        plt.xlabel('Epoch'); plt.ylabel('Loss')
        plt.title(f'{model_name} Loss Curve'); plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(RESULTS_DIR, f'{model_name}_loss.png'), bbox_inches='tight')
        plt.close()

        summary[model_name] = {
            'params': params,
            'best_acc': round(best_acc, 4),
            'final_test_loss': round(history['test_loss'][-1], 4),
            'best_test_loss': round(best_test_loss, 4),
            'time_sec': round(elapsed, 1),
            'history': history,
        }

    # 保存结构化指标，供 analysis.py 使用
    with open(os.path.join(RESULTS_DIR, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 保存人类可读日志
    with open(os.path.join(BASE_DIR, 'train_log.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))

    # 控制台输出对比汇总表
    print('\n' + '=' * 80)
    print(f'{"模型":<14}{"参数量":<10}{"最优测试准确率":<16}{"测试损失":<12}{"训练耗时(s)":<12}')
    print('-' * 80)
    for name, info in summary.items():
        print(f'{name:<14}{info["params"]:<10}{info["best_acc"]:<16.2f}{info["best_test_loss"]:<12.4f}{info["time_sec"]:<12.1f}')
    print('=' * 80)
    print('全部实验完成！权重在 weights/，曲线与指标在 results/。')


if __name__ == '__main__':
    main()
