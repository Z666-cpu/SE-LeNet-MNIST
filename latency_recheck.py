# -*- coding: utf-8 -*-
"""
时延复测脚本：按统一口径（batch=1，每轮预热 20 次、正式测量 200 次，共 5 轮取中位数）
重新加载四个最优权重，测量单样本推理时延，并回写 results/metrics.json。

用途：在不重新训练的前提下，把历史单次平均时延替换为 5 轮中位数口径。
运行方式（在本脚本所在目录）：
    python latency_recheck.py
"""
import os
import json

import torch

from models import LeNet5, SELenet, BNLenet, SEBNLenet
from train_all_models import measure_latency, WEIGHT_DIR, RESULTS_DIR

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
METRICS_PATH = os.path.join(RESULTS_DIR, 'metrics.json')
TRAIN_LOG_PATH = os.path.join(BASE_DIR, 'train_log.txt')

MODEL_FACTORY = {
    'LeNet-5': LeNet5,
    'SE-LeNet': SELenet,
    'BN-LeNet': BNLenet,
    'SE-BN-LeNet': SEBNLenet,
}


def rewrite_train_log(metrics, device):
    """按复测后的 metrics.json 同步重写根目录 train_log.txt，避免日志时延与指标文件不一致。"""
    cfg = metrics['config']
    val_size = cfg['val_size']
    train_size = 60000 - val_size
    lines = [f'设备: {device}', f'种子: {cfg["seeds"]}',
             f'训练 {train_size} / 验证 {val_size} / 测试 10000']
    for name, m in metrics['models'].items():
        lines.append(
            f"{name}: params={m['params']}, "
            f"test_acc={m['test_acc_mean']:.2f}±{m['test_acc_std']:.2f}%, "
            f"test_loss={m['test_loss_mean']:.4f}, "
            f"latency={m['latency_ms']:.3f}ms")
    with open(TRAIN_LOG_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'当前使用设备: {device}')

    with open(METRICS_PATH, 'r', encoding='utf-8') as f:
        metrics = json.load(f)

    for name, model_cls in MODEL_FACTORY.items():
        weight_path = os.path.join(WEIGHT_DIR, f'{name}_best.pth')
        if not os.path.exists(weight_path):
            print(f'跳过 {name}: 权重不存在 {weight_path}')
            continue
        model = model_cls().to(device)
        model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
        latency = measure_latency(model, device)
        latency = round(latency, 4)

        m = metrics['models'][name]
        m['latency_ms'] = latency
        for run in m['runs'].values():
            run['latency_ms'] = latency
        print(f'{name}: 时延中位数 {latency:.4f} ms')

    with open(METRICS_PATH, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print('metrics.json 时延已回写:', METRICS_PATH)

    rewrite_train_log(metrics, device)
    print('train_log.txt 已按 metrics.json 同步:', TRAIN_LOG_PATH)


if __name__ == '__main__':
    main()
