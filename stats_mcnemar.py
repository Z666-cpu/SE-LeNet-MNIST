# -*- coding: utf-8 -*-
"""
统计分析脚本（论文 4.3.3 混淆矩阵易混对 与 4.4 模型对 McNemar 检验）。

前置条件：已运行 train_all_models.py，weights/ 下存在四个最优权重。
运行方式（项目根目录）：
    python stats_mcnemar.py

输出：
    results/stats.json  各模型准确率、Top 易混数字对、三组 McNemar 检验 b/c 与 p 值。

说明：
- McNemar 检验基于同一 10 000 张测试样本的成对预测，判断两模型在该测试集上的
  错误是否存在显著差异；b/c 为“仅 A 错 / 仅 B 错”的样本数。
- 小样本边缘不一致计数下采用二项精确检验（双侧），同时给出带连续性修正的卡方值备查。
"""
import os
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from scipy import stats

from models import LeNet5, SELenet, BNLenet, SEBNLenet

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
WEIGHT_DIR = os.path.join(BASE_DIR, 'weights')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_FACTORY = {
    'LeNet-5': LeNet5,
    'SE-LeNet': SELenet,
    'BN-LeNet': BNLenet,
    'SE-BN-LeNet': SEBNLenet,
}


def build_test_loader():
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_dataset = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)
    return DataLoader(test_dataset, batch_size=256, shuffle=False)


def predict(model_name, model_cls, loader):
    model = model_cls().to(device)
    weight_path = os.path.join(WEIGHT_DIR, f'{model_name}_best.pth')
    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            output = model(data)
            _, pred = torch.max(output, 1)
            preds.append(pred.cpu().numpy())
            targets.append(target.numpy())
    return np.concatenate(preds), np.concatenate(targets)


def top_confusions(cm, top_k=8):
    """返回非对角线上计数最高的易混对（方向：真实 i -> 预测 j）。"""
    pairs = []
    n = cm.shape[0]
    for i in range(n):
        for j in range(n):
            if i != j and cm[i, j] > 0:
                pairs.append({'true': int(i), 'pred': int(j), 'count': int(cm[i, j])})
    pairs.sort(key=lambda x: -x['count'])
    return pairs[:top_k]


def mcnemar(pred_a, pred_b, y_true):
    """模型 A 与 B 的 McNemar 检验。返回 b（A对B错）、c（A错B对）、精确 p、卡方 p。"""
    a_correct = pred_a == y_true
    b_correct = pred_b == y_true
    b = int((a_correct & ~b_correct).sum())   # A 对、B 错
    c = int((~a_correct & b_correct).sum())   # A 错、B 对
    n = b + c
    if n == 0:
        p_exact = 1.0
        p_chi2 = 1.0
    else:
        # 双侧二项精确检验（H0: b 与 c 等概率）
        p_exact = float(stats.binomtest(min(b, c), n, 0.5).pvalue)
        # 带连续性修正的 McNemar 卡方（边缘计数较小时仅作备查）
        chi2 = (abs(b - c) - 1) ** 2 / n if n > 0 else 0.0
        p_chi2 = float(stats.chi2.sf(chi2, 1))
    return {'only_a_correct_b': b, 'only_b_correct_a': c,
            'discordant': n, 'p_exact': round(p_exact, 4), 'p_chi2': round(p_chi2, 4)}


def main():
    loader = build_test_loader()
    preds = {}
    result = {'models': {}, 'mcnemar': {}}

    for name, cls in MODEL_FACTORY.items():
        pred, y_true = predict(name, cls, loader)
        preds[name] = pred
        cm = np.zeros((10, 10), dtype=int)
        for t, p in zip(y_true, pred):
            cm[t, p] += 1
        acc = float((pred == y_true).mean() * 100)
        result['models'][name] = {
            'test_acc': round(acc, 2),
            'top_confusions': top_confusions(cm),
        }
        print(f'{name}: acc={acc:.2f}%  top混淆对(真实->预测:次数): '
              + ', '.join(f"{x['true']}->{x['pred']}:{x['count']}" for x in result['models'][name]['top_confusions']))

    y_true = y_true
    comparisons = [
        ('LeNet-5', 'SE-BN-LeNet'),
        ('BN-LeNet', 'SE-BN-LeNet'),
        ('LeNet-5', 'BN-LeNet'),
    ]
    for a, b in comparisons:
        stat = mcnemar(preds[a], preds[b], y_true)
        result['mcnemar'][f'{a} vs {b}'] = {
            'model_a': a, 'model_b': b, **stat,
        }
        print(f'McNemar {a} vs {b}: A对B错={stat["only_a_correct_b"]}, '
              f'A错B对={stat["only_b_correct_a"]}, 精确p={stat["p_exact"]}, '
              f'卡方p={stat["p_chi2"]}')

    with open(os.path.join(RESULTS_DIR, 'stats.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print('已保存: results/stats.json')


if __name__ == '__main__':
    main()
