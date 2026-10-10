# CWRU 基线对比框架

依据本项目 `docs/` 中 FD-LLM 第 4.1.4 节、第 4.2.1 节及 Table 7，实现
CNN、ResNet18、Transformer、Attention-LSTM、Attention-GRU，以及 SimCLR。
前五种模型按用户要求的顺序开发并逐个完成测试，最后实现 SimCLR。

本实现是**参考文献启发的一维分类适配**，不声称逐层、逐超参数严格复现 FD-LLM
原作者的基线。FD-LLM 只列方法及引文，并未给出这些基线的完整架构配置；其所引
Attention-LSTM/GRU 研究的是 RUL 回归，分类适配需要作出结构选择。CNN 文献使用
物理引导的二维矩形输入与卷积核设计，本项目采用一维宽首层卷积，并未拟合其物理参数。
CNN/两个 attention 引文的完整网络配置未取得，不把下面的默认层数称作原文设置。

## 模型结构与默认参数

统一输入 `(B,1024,1)`，监督分类输出 `(B,19)` 原始 logits。

| 模型 | 结构 | 配置默认值与适配依据 |
| --- | --- | --- |
| CNN | 3 个 Conv1d/BN/ReLU/MaxPool 块 → 全局平均池化 → 分类 | channels=[32,64,128]，kernel=[65,5,5]，stride=[4,1,1]，dropout=0.2。参考 [58] 的振动衰减感受野思路；宽 kernel 是工程值。 |
| ResNet18 | 1D stem → 4 stage × 2 BasicBlock → 全局平均池化 → Linear | 保留 [59] 的 `[2,2,2,2]` 残差布局，stem kernel=7/stride=2，宽度=[64,128,256,512]；2D 改为 1D，不使用 ImageNet 权重。 |
| Transformer | 16 点不重叠 patch → Linear → 正弦位置编码 → 3 层 encoder → 平均池化 → Linear | d_model=128，4 heads，FFN=256，dropout=0.1；参考 [60] 的自注意力与位置编码。64 token patch 化及分类池化是本项目选择，使用 post-norm/ReLU。 |
| Attention-LSTM | 两层 LSTM → additive temporal attention → 加权汇聚 → 分类 | hidden=128，attention=64，dropout=0.1，默认单向；可配置双向。由 [62] 的 attention-LSTM 思路适配为类别预测，注意力权重覆盖全部 1024 步。 |
| Attention-GRU | 两层 encoder GRU → 一步 decoder GRU → query/key additive attention → context/query 拼接 → 分类 | hidden=128，attention=64，dropout=0.1；参考 [63] 的 encoder–decoder attention GRU 思路。一解码查询和分类头是适配选择。 |
| SimCLR | 1D ResNet18 → 两层 MLP projection → NT-Xent 预训练；丢弃 projection → 线性探测 | projection hidden=256/out=128，τ=0.5，默认冻结 backbone；可选 finetune。原 SimCLR 主要采用 ResNet50，本项目默认用已有 ResNet18 便于统一信号处理。 |

全部模型及训练参数位于 `configs/baselines.yaml`，模型类支持关键字参数配置。
各模型结构位于独立子目录，`common.py` 只共享输入契约及分类调用接口。
训练、指标、图表复用 `src/utils/training.py` 和 `src/evaluation/`；通过基线 task/
trainer 包装保存额外的采样器随机状态，不修改任何 FD-LLM 模型或公共代码。

## 数据与公平比较

直接读取已有 `train.npz` 和 `test.npz` 的 X/y；不改变数据、归一化、描述或原始划分。
完整 train=1748、test=437，19 类，每类 115 个窗口，与论文每类 150 个窗口存在差异。
所以后续本项目指标只能在自己的相同条件下比较，不能把它们宣称为 Table 7 的直接复现值。

固定 `split_seed=42`，在训练集内部按类划出 20% 验证集：默认 1406 个训练窗口、342 个
验证窗口。所有基线和重复实验使用相同验证索引，测试集不参与选择 checkpoint。对比
损失预训练也仅优化训练子集，验证子集用于选择最佳预训练 checkpoint，不使用测试样本。

`repeat_seeds=[42,...,51]` 控制每次权重初始化、训练随机性和数据顺序。运行时 YAML 中
`seed` 被当前 repeat_seed 覆盖，`split_seed` 保持不变。训练 DataLoader 使用独立的
Torch Generator，使同种子下不同网络的初始化耗随机数多少不会改变打乱顺序。
worker seed 复用现有工具，默认 num_workers=0。单设备 CPU/CUDA 自动识别，浮点32训练；
不包含 AMP、DDP 或外部预训练权重下载。

每个 run 保存 train/validation/test 索引和源文件 SHA-256。测试集指纹、测试索引、
类别顺序、指标平均方式、smoke/正式模式不同的结果禁止混入同一张表。模型训练超参数
默认统一为 AdamW、lr=1e-3、weight_decay=1e-4、50 epochs、batch=32、grad clip=1。
这些是工程默认值；SimCLR 因为任务不同有单独的预训练配置（100 epochs、batch=64），
分类阶段使用与其他监督网络相同的 training 配置。可用验证集调整配置后另建实验目录。

## SimCLR 的完整流程

1. 对同一窗口独立构造两个视图：随机裁剪到至少 80% 后线性插值恢复 1024 点、幅值缩放
   std=0.1、噪声 std=0.02、连续遮罩比例 5%。这些是时域增强选择，不是原文图像增强参数。
2. feature encoder 和 nonlinear projection 联合训练。NT-Xent 在 2N 个归一化投影上计算，
   正样本为同一窗口的另一个视图，排除自身，其余实例视图为负样本，温度可配置。
   分类标签完全不参与此损失；不把同类别样本扩展成监督正对。
3. 验证增强使用固定独立 generator（validation_seed=12345），每 epoch 重置，降低
   验证随机性。预训练 batch 至少 2；末尾如果只有一个样本，将其合入前一 batch，不丢样本。
4. 根据验证 NT-Xent 选择最佳预训练权重。丢弃 projection，默认 `linear_probe` 冻结
   encoder 参数及 BN 状态，仅训练线性分类器；`finetune` 可联合微调 encoder 和分类器。
5. 使用下游验证交叉熵选择最佳分类 checkpoint，最后在相同测试集计算 19 类指标。

梯度累积不会扩大 NT-Xent 的负样本池；如果显存允许，提高实际 pretraining.batch_size
通常比只增加累积步数更有意义。增强强度需要在验证集检查，当前无正式效果结论。

## 运行方式

从项目根目录运行，依赖沿用现有 `requirements.txt`。

以下为云端正式实验命令（本次未执行）：

```bash
# 单模型、单次；缺省种子为 repeat_seeds 的第一项
python -m baselines.train --model cnn --repeats 1

# 单模型或全部六种模型，每种 10 次独立重复
python -m baselines.train --model simclr --repeats 10
python -m baselines.train --model all --repeats 10

# 改用自己的 YAML；自动选择 CUDA，或在 YAML 中指定 device
python -m baselines.train --model all --config configs/baselines.yaml --output outputs/baselines_experiment_v2

# 增加 YAML 中 epochs 后，从 epoch 边界续训
python -m baselines.train --model cnn --repeats 1 --resume outputs/baselines/cnn/seed_42/training/last.pt
python -m baselines.train --model simclr --repeats 1 --resume-pretraining outputs/baselines/simclr/seed_42/pretraining/last.pt
python -m baselines.train --model simclr --repeats 1 --resume outputs/baselines/simclr/seed_42/training/last.pt

# 只加载分类 checkpoint，重新测试，不训练
python -m baselines.evaluate --checkpoint outputs/baselines/cnn/seed_42/training/best.pt
```

续训保存 optimizer、模型、全局 RNG 和独立 DataLoader Generator 状态；只支持 epoch
边界。除总 epochs 外应保持模型、数据及训练设置不变；不支持一次参数同时恢复多个模型。
SimCLR 下游续训直接恢复 checkpoint 内的 pretrained encoder，不重新预训练。
新运行/改变模型配置应使用新的输出根目录，避免覆盖已有实验。Checkpoint 仅加载可信文件。

本地仅做测试：

```bash
python -m pytest -q
python -m baselines.train --model all --smoke --output outputs/baselines_smoke
python -m baselines.train --model cnn --smoke --repeats 2 --output outputs/baselines_smoke_repeats
```

`--smoke` 强制 CPU、各阶段 1 epoch、缩小网络维度，使用每类 3 个真实训练窗口和
每类 1 个真实测试窗口；输入长度仍为 1024，类别仍为 19。没有改写源数据。此时一共
38 个训练、19 个验证、19 个测试样本。这些更新只是连通性测试，其 accuracy 不能作为
模型性能比较。默认 smoke 只运行 1 个种子，可显式增加用于检查重复实验管道。

## 产物与 Table 7 对比表

```text
outputs/baselines/
  cnn/seed_42/
    config.yaml, split_indices.npz, result.json
    training/  best.pt, last.pt, train.log, history.json, metrics.jsonl, curves.png
    test/      metrics.json, predictions.npz, confusion matrices, t-SNE, curves
  ...其他模型与种子...
  simclr/seed_42/
    pretraining/  预训练 checkpoint、日志与损失曲线
    training/     线性探测或微调 checkpoint 与分类训练记录
    test/         相同测试集评估
  comparison.json, comparison.csv, comparison.md
```

JSON/CSV 保存 0..1 的指标及 `ddof=1` 样本标准差；Markdown 按百分比显示“均值 ± SD”。
仅 1 次时 SD 标记不可估计，不填造一个零标准差。支持 Accuracy、macro Precision/
Recall/F1、weighted Precision/Recall/F1、逐类指标及 19×19 混淆矩阵。汇总表按 Table 7
的顺序排列六个模型，加上 FD-LLM 一行，并扩充 Recall 和重复次数。FD-LLM 尚未正式
实验时保持 Pending，不用论文数值或 Mock 结果填充。论文没有明确多分类指标的平均方式，
本项目统一使用 macro 作为主对比口径；这也是不能直接宣称数值复现的原因之一。

后续全部基线及 FD-LLM 应使用同一份完整测试 NPZ。FD-LLM 的既有评估入口会导出
predictions.npz，可将实际诊断 checkpoint 和评估目录导入同一比较表：

```bash
python -m baselines.comparison --root outputs/baselines
python -m baselines.comparison --root outputs/baselines --fd-checkpoint outputs/diagnosis/best.pt --fd-evaluation outputs/evaluation
```

导入会验证 diagnosis stage、类别顺序、完整测试标签顺序，并以 checkpoint 声明的
train/test 文件计算指纹；从预测重新计算同口径指标，记录 checkpoint SHA-256。
此过程不会加载 LLM，也不能证明所提供的预测确实来自所提供的 checkpoint，调用者需
提供配套产物。不同模型的重复实验应使用同一组种子；FD-LLM 当前内部验证划分随其 seed
变化，基线则固定 split_seed。若正式比较要求相同的内部验证划分，应另行统一 FD-LLM
的划分接口；本次按要求不修改其代码。汇总验证同一外层训练/测试数据，不把内部划分差异
隐瞒为完全相同的训练协议。

## 参考来源

- FD-LLM：本地论文，第 4.1.4、第 4.2.1、Table 7、参考文献 [58]～[63]。
- [58] Ruan et al., *CNN parameter design based on fault signal analysis and its application in bearing fault diagnosis*, 2023：[论文](https://doi.org/10.1016/j.aei.2023.101877)。
- [59] He et al., *Deep Residual Learning for Image Recognition*：[原论文](https://arxiv.org/abs/1512.03385)。
- [60] Vaswani et al., *Attention Is All You Need*：[原论文](https://arxiv.org/abs/1706.03762)。
- [61] Chen et al., *A Simple Framework for Contrastive Learning of Visual Representations*：[公式与评估协议](https://arxiv.org/html/2002.05709v3)。
- [62] Cheng et al., *RUL Prediction Method for Electrical Connectors With Intermittent Faults Based on an Attention-LSTM Model*, 2023：[论文](https://doi.org/10.1109/TCPMT.2023.3282616)。
- [63] Lin et al., *Attention-based Gate Recurrent Unit for remaining useful life prediction in prognostics*, 2023：[论文](https://doi.org/10.1016/j.asoc.2023.110419)。

尚需云端 CUDA 运行、完整数据上的正式 10 次实验、验证集选参及真实 FD-LLM 对比。
当前 CPU 测试仅证明张量流、损失、梯度、冻结、续训、结果导出和统计流程可执行。
