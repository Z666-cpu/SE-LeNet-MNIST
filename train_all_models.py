# -*- coding: utf-8 -*-
"""
一键复现主脚本：训练论文中的全部 4 个模型（基线 + 3 个变体，含消融）。

模型：
  LeNet-5        基线模型（LeNet-5 结构 + 现代训练配置）
  SE-LeNet       仅 SE 通道门控（消融）
  BN-LeNet       仅批归一化（消融）
  SE-BN-LeNet    批归一化 + SE 门控（最终改进模型）

实验规范（针对方法学问题做了修正）：
  1. 从 60000 张训练集中固定划分出 6000 张作为验证集（54000 训练 / 6000 验证），
     测试集 10000 张不参与任何模型选择，仅用于最终评估。
  2. 最优 checkpoint 依据验证集准确率选取（并列时取验证损失更低者），
     报告的测试准确率与测试损失均来自该 checkpoint，口径一致。
  3. 支持多随机种子重复运行，报告均值 ± 标准差，以判断差异是否在随机波动范围内。
  4. 固定随机种子并开启 cuDNN 确定性；测量单样本推理时延。

运行方式（在本脚本所在目录）：
    python train_all_models.py                 # 默认单种子 42
    python train_all_models.py --seeds 42,123,2024   # 多种子重复

输出：
    weights/<模型名>_best.pth      主种子的最优权重
    results/metrics.json           全部模型逐轮指标与多种子汇总
    results/<模型名>_accuracy.png  单模型准确率曲线（主种子）
    results/<模型名>_loss.png      单模型损失曲线（主种子）
    train_log.txt                  人类可读训练日志
"""
import os
import time
import json
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from models import LeNet5, SELenet, BNLenet, SEBNLenet

# ===================== 固定实验配置（与论文一致，请勿随意修改） =====================
EPOCHS = 30
BATCH_SIZE = 64
LR = 0.001
WEIGHT_DECAY = 1e-4
VAL_SIZE = 6000           # 从训练集划分的验证集大小
NUM_WORKERS = 0           # Windows 下建议 0；Linux 可调到 2~4
LATENCY_RUNS = 200        # 推理时延测量次数
# ================================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
WEIGHT_DIR = os.path.join(BASE_DIR, 'weights')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(WEIGHT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def setup_determinism(seed: int):
    """固定全部随机源，开启 cuDNN 确定性，保证同一环境下可复现。"""
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


def split_train_val(base_dataset, val_size, seed):
    """固定种子划分训练 / 验证集，返回两个 Subset。"""
    n = len(base_dataset)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]
    return Subset(base_dataset, train_idx), Subset(base_dataset, val_idx)


def run_epoch(model, loader, criterion, device, train=False, optimizer=None):
    """跑一个 epoch，返回平均损失与准确率（%）。"""
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            if train:
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            else:
                output = model(data)
                loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            _, pred = torch.max(output.data, 1)
            total += target.size(0)
            correct += (pred == target).sum().item()
    return total_loss / total, 100. * correct / total


def measure_latency(model, device, runs=LATENCY_RUNS):
    """测量单样本前向推理时延（毫秒），含 warmup 与 CUDA 同步。"""
    model.eval()
    x = torch.randn(1, 1, 32, 32, device=device)
    with torch.no_grad():
        for _ in range(20):
            model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(runs):
            model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.time()
    return (t1 - t0) / runs * 1000.0


def train_one_model(model_name, model_cls, seed, loaders, device, save_weight=False):
    """训练单个模型（单个种子），返回历史曲线、最优 checkpoint 与最终测试结果。"""
    setup_determinism(seed)
    model = model_cls().to(device)
    params = sum(p.numel() for p in model.parameters())

    train_loader, val_loader, test_loader = loaders
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = {k: [] for k in
               ['train_loss', 'train_acc', 'val_loss', 'val_acc', 'test_loss', 'test_acc']}
    best_val_acc = -1.0
    best_val_loss = float('inf')
    best_epoch = -1
    best_state = None
    t0 = time.time()

    for epoch in range(EPOCHS):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device,
                                    train=True, optimizer=optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device)
        te_loss, te_acc = run_epoch(model, test_loader, criterion, device)

        for k, v in [('train_loss', tr_loss), ('train_acc', tr_acc),
                     ('val_loss', va_loss), ('val_acc', va_acc),
                     ('test_loss', te_loss), ('test_acc', te_acc)]:
            history[k].append(v)

        # 依据验证集选 checkpoint（准确率优先，并列时损失更低）
        if (va_acc > best_val_acc) or (va_acc == best_val_acc and va_loss < best_val_loss):
            best_val_acc = va_acc
            best_val_loss = va_loss
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f'Epoch {epoch + 1:2d}/{EPOCHS} | 训练损失 {tr_loss:.4f} 训练 {tr_acc:.2f}% | '
              f'验证损失 {va_loss:.4f} 验证 {va_acc:.2f}% | 测试损失 {te_loss:.4f} 测试 {te_acc:.2f}%')
        scheduler.step()

    # 加载最优 checkpoint，在测试集上做最终评估（报告口径，准确率与损失一致）
    model.load_state_dict(best_state)
    if save_weight:
        torch.save(best_state, os.path.join(WEIGHT_DIR, f'{model_name}_best.pth'))
    final_test_loss, final_test_acc = run_epoch(model, test_loader, criterion, device)
    latency = measure_latency(model, device)
    elapsed = time.time() - t0

    result = {
        'seed': seed,
        'best_epoch': best_epoch,
        'best_val_acc': round(best_val_acc, 4),
        'test_acc': round(final_test_acc, 4),
        'test_loss': round(final_test_loss, 4),
        'time_sec': round(elapsed, 1),
        'latency_ms': round(latency, 4),
    }
    return params, history, result


def plot_single(model_name, history):
    """绘制单模型准确率 / 损失曲线（主种子）。"""
    epochs = range(1, EPOCHS + 1)
    plt.figure(dpi=200)
    plt.plot(epochs, history['train_acc'], label='Train')
    plt.plot(epochs, history['val_acc'], label='Validation')
    plt.plot(epochs, history['test_acc'], label='Test')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)')
    plt.title(f'{model_name} Accuracy'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, f'{model_name}_accuracy.png'), bbox_inches='tight')
    plt.close()

    plt.figure(dpi=200)
    plt.plot(epochs, history['train_loss'], label='Train')
    plt.plot(epochs, history['val_loss'], label='Validation')
    plt.plot(epochs, history['test_loss'], label='Test')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title(f'{model_name} Loss'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, f'{model_name}_loss.png'), bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=str, default='42',
                        help='逗号分隔的随机种子，如 42 或 42,123,2024')
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    primary_seed = seeds[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'当前使用设备: {device}')
    if torch.cuda.is_available():
        print(f'GPU名称: {torch.cuda.get_device_name(0)}')
    print(f'随机种子: {seeds}')
    print('=' * 60)

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    print('正在加载 MNIST 数据集，首次运行会自动下载...')
    train_base = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)

    # 验证集划分固定使用 primary_seed，保证数据划分本身可复现
    train_subset, val_subset = split_train_val(train_base, VAL_SIZE, primary_seed)
    g = torch.Generator().manual_seed(primary_seed)
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    print(f'训练 {len(train_subset)} 张，验证 {len(val_subset)} 张，测试 {len(test_dataset)} 张')
    print('=' * 60)

    model_factory = {
        'LeNet-5': LeNet5,
        'SE-LeNet': SELenet,
        'BN-LeNet': BNLenet,
        'SE-BN-LeNet': SEBNLenet,
    }
    loaders = (train_loader, val_loader, test_loader)

    metrics = {'config': {
        'epochs': EPOCHS, 'batch_size': BATCH_SIZE, 'lr': LR,
        'weight_decay': WEIGHT_DECAY, 'val_size': VAL_SIZE,
        'seeds': seeds, 'primary_seed': primary_seed,
    }, 'models': {}}
    log_lines = [f'设备: {device}', f'种子: {seeds}',
                 f'训练 {len(train_subset)} / 验证 {len(val_subset)} / 测试 {len(test_dataset)}']

    for model_name, model_cls in model_factory.items():
        runs = {}
        params, primary_history, primary_result = None, None, None
        for i, seed in enumerate(seeds):
            print(f'\n===== {model_name} | seed {seed} | 共 {EPOCHS} 轮 =====')
            is_primary = (seed == primary_seed)
            params, history, result = train_one_model(
                model_name, model_cls, seed, loaders, device,
                save_weight=is_primary)
            runs[str(seed)] = result
            if is_primary:
                primary_history, primary_result = history, result
            print(f'{model_name} seed{seed} 完成: 最优轮 {result["best_epoch"]}，'
                  f'测试 {result["test_acc"]:.2f}%，测试损失 {result["test_loss"]:.4f}，'
                  f'耗时 {result["time_sec"]:.0f}s')

        if primary_history is not None:
            plot_single(model_name, primary_history)

        accs = np.array([runs[str(s)]['test_acc'] for s in seeds])
        losses = np.array([runs[str(s)]['test_loss'] for s in seeds])
        metrics['models'][model_name] = {
            'params': params,
            'primary_seed': primary_seed,
            'history': primary_history,
            'runs': runs,
            'test_acc_mean': round(float(accs.mean()), 4),
            'test_acc_std': round(float(accs.std(ddof=0)), 4),
            'test_loss_mean': round(float(losses.mean()), 4),
            'test_loss_std': round(float(losses.std(ddof=0)), 4),
            'latency_ms': primary_result['latency_ms'],
        }
        log_lines.append(
            f'{model_name}: params={params}, '
            f'test_acc={accs.mean():.2f}±{accs.std(ddof=0):.2f}%, '
            f'test_loss={losses.mean():.4f}, latency={primary_result["latency_ms"]:.3f}ms')

    with open(os.path.join(RESULTS_DIR, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(os.path.join(BASE_DIR, 'train_log.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))

    print('\n' + '=' * 90)
    print(f'{"模型":<14}{"参数量":<9}{"测试准确率(%)":<18}{"测试损失":<11}{"时延(ms)":<10}')
    print('-' * 90)
    for name, m in metrics['models'].items():
        print(f'{name:<14}{m["params"]:<9}{m["test_acc_mean"]:<10.2f}±{m["test_acc_std"]:<6.2f}'
              f'{m["test_loss_mean"]:<11.4f}{m["latency_ms"]:<10.3f}')
    print('=' * 90)
    print('全部实验完成！权重在 weights/，曲线与指标在 results/。')


if __name__ == '__main__':
    main()
