# LeNet-5 手写数字识别复现：批归一化与通道门控的消融研究（MNIST）

> 课程论文复现项目 · 作者：张子泊（华北电力大学 自动化系）
> 论文题目：《LeNet-5 手写数字识别复现：批归一化与通道门控消融研究》

本项目复现经典网络 **LeNet-5**（LeCun et al., 1998）在 MNIST 上的训练流程，
并设置四个配置用于对比与消融：基线 LeNet-5、仅 SE 通道门控、仅批归一化（BN）、SE-BN-LeNet。
实验固定随机种子 `seed=42`，从训练集划分独立验证集选择最优 checkpoint，测试集不参与模型选择；
逐轮测试指标仅用于观察泛化趋势与绘制收敛曲线，最终报告指标来自验证集选出的最优权重。

> **诚实结论提示**：本项目不是"提出更强模型"的改进论文。在 MNIST 这一接近饱和的基准上，
> BN 的主要作用是稳定并加快收敛、降低训练损失；SE 通道门控在本配置下未测得准确率增益，
> BN 与 SE 之间也未观察到协同效应。论文按实验结果如实报告，详见第 4、5 节。

---

## 一、目录结构

```
SE-LeNet-MNIST/
├── models.py              # 四个模型定义：LeNet5 / SELenet / BNLenet / SEBNLenet（含 SEModule）
├── train_all_models.py    # 【主入口】训练全部 4 个模型，划分验证集、按验证集保存最优权重、输出 metrics.json 与单模型曲线
├── analysis.py            # 读取 metrics.json 生成论文对比曲线与基线/改进模型混淆矩阵，并写出自动验收结果
├── stats_mcnemar.py       # 加载四个最优权重，统计易混数字对并完成模型对之间的 McNemar 检验，输出 results/stats.json
├── requirements.txt       # numpy / matplotlib / scipy 依赖（PyTorch 按下方说明单独安装）
├── weights/               # 四个模型在验证集上选出的最优权重（*_best.pth）
├── results/               # 对比曲线、单模型曲线、混淆矩阵、metrics.json、verification.txt
├── train_log.txt          # 人类可读的训练汇总日志
├── data/                  # MNIST 数据集（首次运行自动下载，不入库）
├── word_tmp/              # 论文生成脚本与渲染临时文件（不入库）
└── 论文_LeNet-5手写数字识别复现_批归一化与通道门控消融研究.docx
```

## 二、环境要求

| 项目 | 版本 / 说明 |
|---|---|
| Python | 3.10 及以上（实验环境 3.11.4） |
| PyTorch | 2.6.0（实验环境 2.6.0+cu124） |
| torchvision | 0.21.0（实验环境 0.21.0+cu124） |
| numpy | 1.24.3 |
| matplotlib | 3.7.1 |
| scipy | 1.10.1（McNemar 检验用） |
| GPU | 可选；无 NVIDIA 显卡时代码自动切换 CPU（模型很小，CPU 也可训练，耗时更长） |

## 三、安装依赖

### 1. 创建虚拟环境（推荐）

```bash
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

在本项目根目录下依次执行三条命令：

```bash
# 第一步：训练全部 4 个模型（30 轮，seed=42，固定划分 54000 训练 / 6000 验证）
# 约 35~45 分钟（视 GPU 而定；CPU 约 1~2 小时）
python train_all_models.py --seeds 42

# 第二步：读取 metrics.json 生成论文对比曲线、两张混淆矩阵与自动验收文件
python analysis.py

# 第三步：加载四个最优权重，统计易混数字对并做模型对 McNemar 检验，输出 results/stats.json
python stats_mcnemar.py
```

> 多种子重复（可选，论文为单次运行，无需执行）：
> `python train_all_models.py --seeds 42,123,2024`

运行完成后：

- `weights/`：四个模型在验证集上选出的最优权重 `*_best.pth`
- `results/metrics.json`：全部模型逐轮训练指标与最终汇总（结构化数据）
- `results/accuracy_comparison.png`、`results/loss_comparison.png`：论文图 1、图 2
- `results/confusion_LeNet-5.png`、`results/confusion_SE-BN-LeNet.png`：论文图 3 两张混淆矩阵
- `results/verification.txt`：混淆矩阵对角线准确率与论文表格数字的自动核对结果
- `results/stats.json`：各模型易混数字对与三对模型的 McNemar 检验不一致样本数、p 值
- `train_log.txt`：人类可读训练汇总日志

## 五、实验结果（RTX 3050 Ti Laptop，torch 2.6.0+cu124，seed=42，30 轮，单次运行）

### 5.1 整体性能（论文表 2：基线 vs SE-BN-LeNet）

| 模型 | 参数量 | 测试准确率 | 测试损失 | 单样本时延(ms) |
|---|---|---|---|---|
| LeNet-5（基线） | 61,706 | 99.25% | 0.0308 | 0.687 |
| SE-BN-LeNet | 76,390 | 99.34% | 0.0221 | 1.544 |

### 5.2 消融实验（论文表 3：四个配置，单次运行）

| 模型配置 | 参数量 | 测试准确率 | 测试损失 |
|---|---|---|---|
| LeNet-5（基线） | 61,706 | 99.25% | 0.0308 |
| +SE（仅通道门控） | 76,106 | 99.25% | 0.0308 |
| +BN（仅批归一化） | 61,990 | 99.33% | 0.0195 |
| +BN+SE（SE-BN-LeNet） | 76,390 | 99.34% | 0.0221 |

### 5.3 开销对比（论文表 4）

| 模型 | 参数量 | MACs | 单样本时延(ms) |
|---|---|---|---|
| LeNet-5（基线） | 61,706 | 416,520 | 0.687 |
| SE-LeNet | 76,106 | 430,920 | 0.922 |
| BN-LeNet | 61,990 | 416,520 | 0.821 |
| SE-BN-LeNet | 76,390 | 430,920 | 1.544 |

### 5.4 结果解读（与论文一致）

1. **BN 的主要贡献是改善训练过程**：含 BN 模型训练初期收敛更快、曲线更平稳，测试损失明显更低
   （BN-LeNet 0.0195，基线 0.0308）；但在 MNIST 上最终测试准确率的差异很小，McNemar 检验
   三对模型的 p 值均大于 0.05（基线 vs SE-BN-LeNet p=0.349，BN-LeNet vs SE-BN-LeNet p=1.000，
   基线 vs BN-LeNet p=0.416），错误差异不具统计显著性。
2. **SE 在本配置下未测得准确率增益**：+SE 与基线准确率、损失完全相同（99.25% / 0.0308）。
   原因是 SE 被置于输出空间尺寸为 1×1 的 C5 特征之后，全局平均池化退化为恒等映射，
   SE 实质只是对 120 维向量的两层 MLP 门控。
3. **未观察到 BN 与 SE 的协同效应**：BN+SE（99.34%）与仅 BN（99.33%）几乎相同，
   SE 额外引入约 1.47 万个参数（参数量增加约 23.8%）并带来一定时延上升，未换来准确率收益。

## 六、模型结构

- **LeNet-5（基线）**：C1(6×5×5) → ReLU → MaxPool → C3(16×5×5) → ReLU → MaxPool → C5(120×5×5) → ReLU → FC(84) → Dropout(0.25) → FC(10)
- **SE-LeNet**：在基线 C5 后加入 SE 通道门控模块（压缩比 r=2）
- **BN-LeNet**：每个卷积层后加入 `BatchNorm2d`
- **SE-BN-LeNet**：每个卷积层后加入 `BatchNorm2d`，C5 后加入 SE 模块

**统一训练配置**：Adam 优化器，初始学习率 0.001，权重衰减 1e-4，批次大小 64，训练 30 轮，
CosineAnnealingLR 学习率调度，Dropout 0.25；输入缩放为 32×32，归一化均值 0.1307、标准差 0.3081。
从 60 000 张训练图像中固定划分 54 000 训练 / 6 000 验证；10 000 张测试集不参与模型选择，
逐轮测试指标仅用于观察泛化趋势，最终报告指标来自验证集选出的最优权重。

> 说明：本文复现的是"LeNet-5 结构 + 现代训练配置"的强基线（使用 ReLU、最大池化、Dropout 与缩放插值），
> 并非 1998 年原始 LeNet-5 的训练配方。

## 七、复现性说明（重要）

1. 脚本固定 `random`、NumPy、PyTorch（CPU/CUDA）随机种子为 42，并开启
   `cudnn.deterministic=True`，保证同一环境下重复运行结果可复现。
2. 最优 checkpoint 依据**验证集**准确率选取（并列时取验证损失更低者），报告的测试准确率与测试损失
   均来自该 checkpoint，口径一致；测试集不参与模型选择。
3. 本项目为**单次运行**（一个种子），未在多种子上重复，因此不就跨种子波动幅度作定量结论。
   模型对准确率差异是否显著，在同一测试集上以 McNemar 检验评估（见 `stats_mcnemar.py` 与 `results/stats.json`）；
   本项目三对模型 p 值均大于 0.05，差异不显著。不同 PyTorch / CUDA / cuDNN 版本、不同显卡之间浮点运算
   顺序存在细微差异，训练曲线趋势与上述结论保持一致即视为复现成功。若需统计层面更强的结论，
   应使用多种子重复并报告均值±标准差。
4. `weights/` 内附带对应环境训练好的最优权重，可直接由 `analysis.py` 加载复现混淆矩阵与准确率，无需重新训练。

## 八、常见问题

**Q：MNIST 下载失败 / 很慢？**
A：脚本默认从 torchvision 官方源下载。若网络不通，可手动将 MNIST 数据集放入 `data/MNIST/raw/`
（`train-images-idx3-ubyte.gz` 等四个文件），再运行脚本，脚本检测到本地数据后不会重复下载。

**Q：`torch.load(...)` 报错 `weights_only`？**
A：本项目 `analysis.py` 已使用 `weights_only=True`；PyTorch 1.13 以下版本不支持该参数，请升级到 PyTorch 2.x。

**Q：没有 GPU 能不能跑？**
A：可以。脚本自动检测 `cuda`，不可用时使用 CPU；MNIST 与本模型规模很小，CPU 30 轮约 1~2 小时。

## 九、参考文献

1. LeCun Y, Bottou L, Bengio Y, et al. Gradient-based learning applied to document recognition[J]. Proceedings of the IEEE, 1998, 86(11): 2278-2324.
2. Krizhevsky A, Sutskever I, Hinton G E. ImageNet classification with deep convolutional neural networks[J]. Communications of the ACM, 2017, 60(6): 84-90.
3. Simonyan K, Zisserman A. Very deep convolutional networks for large-scale image recognition[J/OL]. arXiv:1409.1556, 2014.
4. He K, Zhang X, Ren S, et al. Deep residual learning for recognition[C]//CVPR. 2016: 770-778.
5. Ioffe S, Szegedy C. Batch normalization: Accelerating deep network training by reducing internal covariate shift[C]//ICML. PMLR, 2015, 37: 448-456.
6. Hu J, Shen L, Sun G. Squeeze-and-excitation networks[C]//CVPR. 2018: 7132-7141.
7. Woo S, Park J, Lee J Y, et al. CBAM: Convolutional block attention module[C]//ECCV. 2018: 3-19.
8. LeCun Y, Cortes C, Burges C J. MNIST handwritten digit database[EB/OL]. http://yann.lecun.com/exdb/mnist/.
9. 段莉莉, 章磊. 一种基于改进型卷积神经网络的手写数字识别方法[J]. 现代电子技术, 2021, 44(18): 107-110.
10. 李波燕, 张勇, 袁德荣, 等. 基于注意力机制的手写体数字识别[J]. 计算机科学, 2022, 49(11A): 211100009-5.
11. 牟世桂. 基于卷积神经网络的 MNIST 手写数字识别优化研究[J]. 计算机应用文摘, 2025(3): 81-83,86.
12. Iwana B K, Kusuda A. Vision Conformer: Incorporating convolutions into vision transformer layers[C/OL]//ICDAR. 2023. arXiv:2304.13991.
