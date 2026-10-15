# Attention 与 MHA / MQA / GQA

Attention 是 Transformer 里负责“从其他位置取信息”的模块。本专题从缩放点积注意力开始，逐步讲到多头注意力、KV cache，以及 MHA、MQA、GQA 三种 Q/K/V 头组织方式。

## 阅读顺序

1. [缩放点积注意力](01-缩放点积注意力.md)：Q/K/V、Softmax、mask 和矩阵形状。
2. [多头注意力](02-多头注意力.md)：为什么要分头，head dimension 和参数量如何计算。
3. [MHA、MQA、GQA](03-MHA-MQA-GQA.md)：共享 K/V 的结构、KV cache 内存和广播关系。
4. [训练与推理实现](04-训练与推理实现.md)：prefill/decode、KV cache、FlashAttention、padding 和常见 bug。
5. [面试题与推导](05-面试题与推导.md)：复杂度、位置编码衔接、因果 mask 和高频追问。

[舒适阅读版](阅读版.html) 把章节连成一篇。它属于 `notes/architecture`，因为这里讨论的是模型结构；训练集群、部署和监控请放到 `infra/`。

## 约定

文中默认讨论 decoder-only Transformer 的自注意力，除非另有说明。`L` 表示序列长度，`d_model` 表示隐藏维度，`h` 表示 query head 数，`d_h=d_model/h` 表示每个 query head 的维度，`g` 表示 GQA 的 KV 头数。

示例矩阵只用于说明形状，不代表某个具体模型的配置。图是 Python 绘制的教学示意，不是性能测量。

## 文件说明

- `assets/`：简单结构图和内存对比图。
- `src/build.py`：绘图、公式和阅读版生成脚本。
- `src/requirements.txt`：构建阅读版所需依赖。

返回 [资料总入口](../../../README.md)。
