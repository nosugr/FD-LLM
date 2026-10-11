# AutoDL 部署与检查

运行目标为 Linux、conda `fdllm`（Python 3.9）和单张 RTX 4090。代码通过
PyCharm SSH 同步；本地 Python、磁盘路径和历史 CPU smoke 结果不代表服务器环境。

## SSH 解释器及同步

当前 `.idea/misc.xml` 保存的是 `/root/miniconda3/bin/python3`，属于 base 路径。
在 PyCharm 的 Python Interpreter 设置中选择服务器 `fdllm` 的解释器，通常为
`/root/miniconda3/envs/fdllm/bin/python`；以服务器 `conda run -n fdllm python -c
"import sys; print(sys.executable)"` 输出为准。仅修改项目 XML 的 SDK 名称无法注册
新的远程解释器，因此没有替换该文件。

当前 deployment 映射为项目根目录到 `/root/autodl-tmp/demo`。运行配置的远程工作
目录设为该目录，并以 module name 执行 `src.alignment.train_alignment` 等入口，
不要直接执行 `src/.../train_alignment.py`。确认同步目录内直接包含 `src/`、
`baselines/` 和 `configs/`，避免多套一层 `FD-LLM/`。

模型和数据被 `.gitignore` 排除，PyCharm 上传也可能有单独的排除规则。必须确认服务器
有 `models/vicuna-7b-v1.5/` 的全部权重分片及 tokenizer，以及
`data/datasets/CWRU/processed/{train.npz,test.npz,class_descriptions.json}`。
也可以在 YAML 中指定服务器绝对路径，例如 `/root/autodl-tmp/models/vicuna-7b-v1.5`。
配置中的相对数据、模型、输出路径基于代码项目根目录解析；Linux 路径大小写须准确。
服务器配置不得引用 `D:\...` 或 Windows 反斜杠路径。

## Python 3.9 依赖

先检查已有环境，不要依据本地安装结果判断。以下安装命令仅在需要配置服务器环境时执行：

```bash
conda activate fdllm
cd /root/autodl-tmp/demo
python --version
python -c "import sys; print(sys.executable)"
nvidia-smi
# 根据驱动选择官方 CUDA wheel；以下是 CUDA 12.4 示例。
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt -c constraints-autodl.txt
python -m pip check
export MPLBACKEND=Agg
python -m src.utils.check_server
```

PyTorch 2.6.0 的 CUDA 11.8、12.4、12.6 wheel 见
[官方安装表](https://pytorch.org/get-started/previous-versions/#v260)。根据服务器驱动兼容性
选择，不以本地 CUDA 或单独安装的 CUDA toolkit 版本推断可用性。
约束文件固定一组支持 Python 3.9 的直接依赖版本；它不是全部传递依赖的锁文件，
也不代表这些版本已在该服务器完成实测。无需 torchvision、torchaudio、flash-attn
或 bitsandbytes；代码使用 eager attention 和未量化的 Vicuna。

预检会核对解释器、依赖导入、CUDA 运算、float32 LSTM 反向、数据输入形状、模型分片、
tokenizer 和关键模块导入。它不下载模型、不写训练产物、不加载完整 7B 权重。
通过后再按 `docs/implementation.md` 运行两个阶段训练及评估。

## GPU 与历史产物

正式 YAML 默认 `device: cuda`，CUDA 不可用时立即失败；测试配置可显式使用 CPU。
4090 的实际 dtype 通过 PyTorch 检测，默认优先 bfloat16，LSTM 保持 float32。
第二阶段默认 batch=1、梯度累积=8、非重入梯度检查点；7B 半精度权重约占 13 GiB，
还需激活、梯度和优化器空间。24 GiB 是否足够须以真实模型首次运行的峰值为准；
评估默认 batch=4、候选块=4，会展开为最多 16 条序列，若 OOM 可先将两者均降至 1。
第一阶段文本编码可降低 `alignment.text_batch_size`。

父目录 `smoke/` 等旧产物含 Windows 绝对路径和随机微型模型配置，不应当作正式服务器
配置或真实 Vicuna checkpoint 使用。不要直接迁移这些配置来续训。正式 checkpoint
默认使用相对配置路径；`torch.load` 已指定 CPU 映射，随后迁移到目标 CUDA 设备。

现有 `tests/` 在项目父目录，不在当前 SSH 项目映射内。README 中原先从 FD-LLM 根目录
运行测试的命令不适用于这个目录布局。若要运行回归测试，需要同步父目录的 tests 到
服务器项目的 tests 目录，或单独设置 PYTHONPATH；服务器预检不依赖这些测试文件。
