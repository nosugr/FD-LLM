# FD-LLM

基于论文第 3 章与 Fig. 1 的 CWRU 两阶段故障诊断代码框架。
默认模型为配置指定的 Vicuna-7B；本地 CPU 使用离线微型模型测试，不下载权重。

- 第一阶段：LSTM 数据编码器、冻结文本编码器、双向多正样本对比学习。
- 第二阶段：模糊语义嵌入、连续与离散 Prompt、Vicuna + LoRA、答案 token 交叉熵。
- 评估：19 类指标、混淆矩阵、训练曲线、t-SNE 与预测结果导出。

配置入口：`configs/fd_llm.yaml`。部署命令、论文与实现选择的区别、checkpoint 和
离线测试说明见 [实现文档](docs/implementation.md)。

从项目根目录运行离线验证：

```bash
python -m pytest -q
python -m tests.smoke_fd_llm --output outputs/smoke
```

本项目当前已实现可测试框架，尚未正式训练或验证真实 Vicuna 的诊断精度。
