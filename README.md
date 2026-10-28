# SE-BN-LeNet：基于通道注意力与批归一化的 LeNet-5 改进（MNIST 手写数字识别）

> 课程论文复现项目 · 作者：张子泊（华北电力大学 自动化系）
> 论文题目：《基于通道注意力与批归一化的 LeNet-5 手写数字识别改进研究》

本项目完整复现经典论文 **LeNet-5**（LeCun et al., 1998）在 MNIST 上的训练过程，
并实现四个模型用于对比与消融：基线 LeNet-5、仅 SE 通道注意力、仅批归一化、SE-BN-LeNet（最终改进模型）。
所有实验固定随机种子 `seed=42`，超参数、数据预处理与论文完全一致。

---

## 一、目录结构

```
SE-LeNet-MNIST/
├── models.py              # 四个模型定义：LeNet5 / SELenet / BNLenet / SEBNLenet（含 SEModule）
├── train_all_models.py    # 【主入口】一键训练全部 4 个模型，输出权重、指标 JSON、训练日志与单模型曲线
├── analysis.py            # 读取 metrics.json 生成论文对比曲线，并基于权重绘制混淆矩阵
├── requirements.txt       # numpy / matplotlib 依赖（PyTorch 需按下方说明单独安装）
├── weights/               # 训练好的最优模型权重（.pth）
├── results/               # 对比曲线、单模型曲线、混淆矩阵、metrics.json
├── logs/                  # 开发过程中的历史训练日志（归档，复现不依赖）
├── data/                  # MNIST 数据集（首次运行自动下载）
└── 论文_基于通道注意力与批归一化的LeNet-5手写数字识别改进研究.docx
```

## 二、环境要求

| 项目 | 版本 / 说明 |
|---|---|
| Python | 3.10 及以上（实验环境 3.11.4） |
| PyTorch | 2.6.0（实验环境 2.6.0+cu124） |
| torchvision | 0.21.0（实验环境 0.21.0+cu124） |
| numpy | 1.24.3 |
| matplotlib | 3.7.1 |
| GPU | 可选；无 NVIDIA 显卡时代码自动切换 CPU（小模型 CPU 也可训练，耗时更长） |

## 三、安装依赖

### 1. 创建虚拟环境（推荐）

```bash
# conda
conda create -n lenet python=3.11 -y
conda activate lenet
```

### 2. 安装 PyTorch（二选一）

**有 NVIDIA GPU（CUDA 12.4，与实验环境一致）：**

```bash
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

**无 GPU（CPU 版本）：**

```bash
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
```

### 3. 安装其余依赖

```bash
pip install -r requirements.txt
```

## 四、一键复现步骤

在本项目根目录下依次执行两条命令：

```bash
# 第一步：训练全部 4 个模型（30 轮，seed=42），约 20~40 分钟（视 GPU 而定）
python train_all_models.py

# 第二步：生成论文对比曲线与混淆矩阵
python analysis.py
```

运行完成后：

- `weights/`：四个模型的最优权重 `*_best.pth`
- `results/metrics.json`：全部模型逐轮训练指标（结构化数据）
- `results/accuracy_comparison.png`、`results/loss_comparison.png`：论文图 1、图 2
- `results/confusion_LeNet-5.png`、`results/confusion_SE-BN-LeNet.png`：混淆矩阵（论文图 3）
- `train_log.txt`：人类可读训练日志

## 五、实验结果（实验环境：RTX 3050 Ti Laptop，torch 2.6.0+cu124，seed=42，30 轮）

### 5.1 主对比（论文表 2）

| 模型 | 参数量 | 测试准确率 | 测试损失 | 训练耗时 |
|---|---|---|---|---|
| LeNet-5（基线） | 61,706 | 99.30% | 0.0315 | 608 s |
| SE-BN-LeNet（改进） | 76,390 | **99.37%** | **0.0206** | 588 s |

### 5.2 消融实验（论文表 3）

| 模型配置 | 参数量 | 最优测试准确率 |
|---|---|---|
| LeNet-5（基线） | 61,706 | 99.30% |
| +SE（仅通道注意力，深层 120 通道） | 76,106 | 99.28% |
| +BN（仅批归一化） | 61,990 | 99.38% |
| +BN+SE（SE-BN-LeNet） | 76,390 | 99.37% |

**结论（与论文一致）**：批归一化是训练加速与性能提升的主要来源；SE 通道注意力在 MNIST 这类简单任务上增益有限，与 BN 组合后性能与仅 BN 相当（差异在随机波动范围内），其通道重标定价值在更复杂任务中更易体现。

## 六、模型结构

- **LeNet-5（基线）**：C1(6×5×5) → ReLU → MaxPool → C3(16×5×5) → ReLU → MaxPool → C5(120×5×5) → ReLU → FC(84) → Dropout(0.25) → FC(10)
- **SE-LeNet**：在基线 C5 后加入 SE 通道注意力模块（压缩比 r=2）
- **BN-LeNet**：每个卷积层后加入 `BatchNorm2d`
- **SE-BN-LeNet**：每个卷积层后加入 `BatchNorm2d`，C5 后加入 SE 模块

**统一训练配置**：Adam 优化器，初始学习率 0.001，权重衰减 1e-4，批次大小 64，30 轮，
CosineAnnealingLR 学习率调度，Dropout 0.25；输入缩放为 32×32，归一化均值 0.1307、标准差 0.3081。

## 七、复现性说明（重要）

1. 脚本已固定 `random`、NumPy、PyTorch（CPU/CUDA）随机种子为 42，并开启
   `cudnn.deterministic=True`、`cudnn.benchmark=False`，保证同一环境下重复运行结果一致。
2. 不同 PyTorch / CUDA / cuDNN 版本、不同显卡之间，浮点运算顺序存在细微差异，
   测试准确率通常会在论文数值 **±0.1 个百分点** 范围内波动，训练曲线趋势与结论保持一致即视为复现成功。
3. 论文中报告的绝对数值来自上述实验环境；`weights/` 内已附带对应环境训练好的权重，
   可直接用 `analysis.py` 加载复现混淆矩阵与准确率，无需重新训练。

## 八、常见问题

**Q：MNIST 下载失败 / 很慢？**
A：脚本默认从 torchvision 官方源下载。若网络不通，可手动将 MNIST 数据集放入 `data/MNIST/raw/`
（`train-images-idx3-ubyte.gz` 等四个文件），再运行脚本，脚本检测到本地数据后不会重复下载。

**Q：`torch.load(...) 报错 weights_only`？**
A：本项目 `analysis.py` 已使用 `weights_only=True`；PyTorch 1.13 以下版本不支持该参数，
请升级到 PyTorch 2.x。

**Q：没有 GPU 能不能跑？**
A：可以。脚本自动检测 `cuda`，不可用时使用 CPU；MNIST 与本模型规模很小，CPU 30 轮约 1~2 小时。

## 九、参考文献

1. LeCun Y, Bottou L, Bengio Y, et al. Gradient-based learning applied to document recognition[J]. Proceedings of the IEEE, 1998, 86(11): 2278-2324.
2. Krizhevsky A, Sutskever I, Hinton G E. ImageNet classification with deep convolutional neural networks[J]. Communications of the ACM, 2017, 60(6): 84-90.
3. Simonyan K, Zisserman A. Very deep convolutional networks for large-scale image recognition[J]. arXiv:1409.1556, 2014.
4. He K, Zhang X, Ren S, et al. Deep residual learning for image recognition[C]//CVPR. 2016: 770-778.
5. Ioffe S, Szegedy C. Batch normalization: Accelerating deep network training by reducing internal covariate shift[C]//ICML. PMLR, 2015, 37: 448-456.
6. Hu J, Shen L, Sun G. Squeeze-and-excitation networks[C]//CVPR. 2018: 7132-7141.
7. Woo S, Park J, Lee J Y, et al. CBAM: Convolutional block attention module[C]//ECCV. 2018: 3-19.
8. LeCun Y, Cortes C, Burges C J. MNIST handwritten digit database. http://yann.lecun.com/exdb/mnist/
