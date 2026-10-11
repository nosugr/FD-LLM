# FD-LLM 实现与部署

本实现依据 `docs/FD-LLM Large language model for fault diagnosis of complex equipment.pdf`
第 3 章、Fig. 1，以及公式 (3)～(10)。当前只完成工程框架和离线接口验证，
真实 Vicuna 权重的服务器加载、正式训练与诊断精度仍需在目标环境验证。

## 结构与论文对应

| 模块 | 实现 | 论文对应 / 本项目选择 |
| --- | --- | --- |
| 数据编码器 | `src/alignment/encoders/data_encoder.py` 复用 `lstm.py` | LSTM 对应式 (1)；默认 2 层、256 hidden、层间 dropout 0.1，取最后状态，经 Linear 投影到 4096。这些细节是工程默认值。 |
| 文本编码器 | `text_encoder.py` | 加载 Vicuna 的无 LM head backbone，冻结并保持 eval。最后一层 hidden states 的有掩码平均池化为默认选择，可改 last pooling；论文未指定池化。 |
| 对比损失 | `contrastive_loss.py` | cosine、τ=0.07、ε=1e-6 和双向均值对应式 (3)～(6)。默认固定 τ，可选训练其对数；是否训练 τ 论文未明示。 |
| 模糊嵌入 | `diagnosis/fuzzy/embedding.py` | 式 (7)～(9)：cosine → **无温度缩放**的 softmax → 原始文本向量加权和。 |
| Prompt | `diagnosis/prompt/learner.py` | 第 3.2 节模板顺序：Human/Data 分隔符 → 数据向量 → 连续 Prompt → Fuzzy 分隔符及向量 → 诊断问题 → Assistant。8 个连续 token、正态初始化 std=0.02 为默认选择。 |
| Vicuna + LoRA | `diagnosis/llm/backbone.py` | 保持基座冻结，仅 q_proj/v_proj 的 LoRA 参数可训练；r=8、alpha=16、dropout=0.05 为默认选择。 |
| 诊断目标 | `diagnosis/model.py` | 式 (10) 的自回归 token 交叉熵。提示、模态 token 和 padding 标记为 -100；只监督答案 token（含 EOS）。 |
| 19 类评估 | `evaluation/` | 以每个类的标准答案平均 token 对数似然作为类别分数，不增加分类头；这是为可重复闭集评估选择的解码方式。另提供自由文本 generate。 |

输入数据维度为 `(B,1024,1)`；真实 Vicuna 配置下数据、文本及 Prompt 的 embedding
维度均为 4096。工厂加载真实权重后会检查 hidden_size，一旦不匹配立即报错。

## 多正样本与特征库

现有 19 类各有两条描述，第一阶段将 38 条描述都编码并缓存。对比损失正样本掩码为
`data_class == text_class`，分子使用同类别全部正样本的 `logsumexp`，避免共享文本成为
错误负样本。每行仅一个正样本时严格退化为论文式 (4)～(6)。这是对原式的多正样本
扩展，并非论文逐式给出的新公式。文本→数据方向忽略当前 minibatch 中没有正样本的
文本行；数据→文本方向要求每个样本都有对应类文本。单一类别 batch 的对比约束较弱，
默认 batch=32；GPU 上可减小 batch，但梯度累积不会扩大对比负样本池。

第二阶段对同类文本特征取平均，按类 ID 顺序形成 19 个 prototype。先聚合再套用式
(7)～(9) 是本项目对多描述文本的选择。训练与推理复用第一阶段缓存，避免同时加载两份
7B 权重；文本编码器保持冻结。第二阶段数据编码器冻结且强制 eval，连续 Prompt 和
LoRA 可训练。用于 fuzzy 加权和的文本 prototype 保持原始幅值。

## 数据与划分

直接读取既有 NPZ 中的 `X`、`y`，不重新采样、归一化或生成描述。现有数据是训练集
拟合的全局 Z-score；第 3.1 节提及最大值归一化，因此预处理方法存在差异。本次按用户
要求沿用已有产物，避免更改数据。需要严格复现时应另行研究并重建预处理产物。

从 `train.npz` 按类别使用固定 seed 划出 20% 验证集，不使用 `test.npz` 选择 checkpoint。
当前 train=1748，test=437，19 类。描述中的 class_id 必须为 0..18，并与数据匹配。
划分是现有训练窗口上的分层划分，不能代替跨设备或跨工况的泛化实验。

## 配置与云端运行

两个模型阶段的所有配置集中在 `configs/fd_llm.yaml`。CWRU 预处理、描述文本生成和
可视化共用带注释的 `configs/cwru.yaml`；旧 LSTM JSON 为兼容保留。
**训练入口以 YAML 为准**。模块导入路径沿用 `src.*`，从项目根目录运行。

目标环境为 AutoDL Linux、conda `fdllm`（Python 3.9）、RTX 4090；SSH 解释器、同步、
依赖安装和服务器预检见 [AutoDL 指南](autodl.md)。历史本地 CPU 验证不代表服务器实测。
本框架支持单 CPU / 单 CUDA 设备；不包含 DDP 或 4-bit 量化。

1. 将项目、处理好的 CWRU 数据和下载完成的 Vicuna 模型部署到云端。
2. `configs/fd_llm.yaml` 已配置为项目内的 `models/vicuna-7b-v1.5`；也可以将
   `model.name_or_path` 改为其他本地完整权重目录或 Hugging Face ID。默认
   `local_files_only: true` 阻止网络下载。仅在确实允许联网下载时显式改为 false。
   真实入口不会自动构造随机模型或降级到测试替身。
3. 正式配置 `device: cuda`，GPU 不可用即报错；显式配置 `auto` 可自动选择 CUDA/CPU。
   dtype auto 在 CPU 用 float32，在支持的 CUDA
   用 bfloat16，否则 float16。LSTM 与缓存特征保持 float32，送入 LLM 时转换到
   embedding dtype。LoRA 和 Prompt 使用可训练的浮点参数。
4. 不同显存可调整 batch_size、梯度累积次数和 gradient_checkpointing；默认启用
   第二阶段非重入激活 checkpoint。7B 权重仍需足够显存，未实测 GPU 显存用量。

```bash
python -m src.alignment.train_alignment --config configs/fd_llm.yaml
python -m src.alignment.train_alignment --config configs/fd_llm.yaml --resume outputs/alignment/last.pt

python -m src.diagnosis.train_diagnosis --config configs/fd_llm.yaml
python -m src.diagnosis.train_diagnosis --config configs/fd_llm.yaml --resume outputs/diagnosis/last.pt

python -m src.evaluation.evaluate --config configs/fd_llm.yaml --stage alignment --checkpoint outputs/alignment/best.pt
python -m src.evaluation.evaluate --config configs/fd_llm.yaml --checkpoint outputs/diagnosis/best.pt
```

评估两个阶段时可分别修改 evaluation.output_dir，避免覆盖各自报告。

## Checkpoint、日志与续训

每个 epoch 完成训练及验证后，原子保存 `last.pt`，验证 loss 改善时保存 `best.pt`。
checkpoint 包含 epoch、最佳 loss、完整配置、历史指标、optimizer 和 Python/NumPy/
Torch/CUDA RNG 状态。支持 **epoch 边界**续训；不支持从中断的半个 batch/epoch 接续。
默认 num_workers=0 的 CPU 测试验证了中断续训与连续运行逐位一致，CUDA 上不能保证
跨硬件逐位一致。梯度累积按组内实际 batch 数平均；尾组也会更新并进行梯度裁剪。

第一阶段 checkpoint 保存数据编码器、对比温度及文本库。续训不需重新加载文本模型。
第二阶段保存 LoRA adapter、连续 Prompt、冻结数据编码器、文本 prototype、类顺序及
答案映射，**不重复保存 7B 基座**。重载需原始基座和 tokenizer，但不再依赖第一阶段
checkpoint；第二阶段初次启动必须有第一阶段 best.pt。续训会核对模型及数据配置，
可增加总 epochs；变更结构或标签映射时应新建实验目录。
保存描述 JSON 的 SHA-256 指纹，防止同路径、同类 ID 下修改文本后仍误用旧缓存。

训练产物包括 `train.log`、`metrics.jsonl`、`history.json` 和 `curves.png`。
第一阶段记录 loss / top1；第二阶段记录答案 token loss / token accuracy，验证另记录
19 类 diagnosis accuracy。Loss/token accuracy 按有效答案 token 数聚合，类别准确率按
样本聚合。Checkpoint 使用 torch 序列化，应仅加载可信来源文件。

## 推理与评估

```python
import torch
from src.utils.runtime import load_config, select_device
from src.diagnosis.train_diagnosis import load_diagnosis

config = load_config()
device = select_device(config["device"])
model = load_diagnosis(config, "outputs/diagnosis/best.pt", device)
signals = torch.randn(2, 1024, 1, device=device)  # 替换为同样预处理的真实输入
result = model.predict(signals, candidate_chunk_size=4)
class_ids = result["class_ids"]
answers = model.generate(signals, max_new_tokens=96)
```

类别分数不是校准概率；predict 无需标签，按生成式模型对完整候选答案的似然评分。
EOS 按有效答案 token 监督，即使 pad_token 与 EOS 相同也不会误屏蔽。默认 canonical
answer 为 `The bearing state is {class_name}.`，并非论文提供的具体 CWRU 答案原文；
可通过 YAML 模板修改。自由文本生成可能产生不在 19 类映射中的答案，闭集指标采用
候选评分，不对自由文本做未经说明的类别解析。

输出包含 Accuracy、macro/weighted Precision、Recall、F1、逐类指标（无预测时按 0
处理）、19×19 混淆矩阵、归一化混淆矩阵、预测 NPZ、训练曲线及 t-SNE PNG/坐标。
t-SNE 先用至多 50 维 PCA 降维、固定 seed、自动调整小样本 perplexity，导出抽样索引；
可视化的是数据编码器特征，不将其分布直接解释为 LLM 推理能力或诊断准确率。

## 离线验证

以下命令仅在 `tests/` 已同步至服务器项目根目录时可用；当前工作区的测试位于项目父目录。

```bash
python -m pytest -q
python -m tests.smoke_fd_llm --output outputs/smoke
```

测试替身只存在于 `tests/`：随机初始化一层、hidden_size=16 的微型 Llama 和本地
WordLevel tokenizer，使用真实 Transformers/PEFT 接口。端到端 smoke 从已有 CWRU
训练集每类取 2 个真实 1024 点窗口，并从原测试集每类取 1 个窗口，两个阶段各运行
一个验证 epoch，然后重载 checkpoint、输出 19 类评分和生成文本、导出评估文件。
这些少量更新仅用于测试，不是正式模型训练；随机微型模型的 accuracy 没有性能意义。
测试不会访问网络或下载任何模型权重。

仍需云端真实 Vicuna 的首次加载、显存及 CUDA 测试、两阶段正式训练和效果验证。
已通过的 CPU 测试只能证明公式、张量流、冻结/梯度和持久化接口可执行。
