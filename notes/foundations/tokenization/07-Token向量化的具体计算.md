# 07 · Token 向量化：从 ID 到 Transformer 输入

分词结束后得到的是整数 ID，例如 `[2, 5, 1]`。模型不能把整数编号当作有大小关系的连续特征直接使用：ID=5 并不比 ID=2 “更像”某个词。

下一步是 **embedding lookup（嵌入查表）**：把每个 ID 映射成一个可学习的稠密向量。它看起来像矩阵乘法，但实现上通常是按行索引，避免真的把 one-hot 向量完整构造出来。

## 1. 先准备一个能手算的例子

假设词表有 6 个 token，模型隐藏维度 `d_model=4`。embedding 矩阵 `E` 的形状是 `[6,4]`：

$$
E=
\begin{bmatrix}
0.1&0.2&0.3&0.4\\
0.5&0.6&0.7&0.8\\
0.9&1.0&1.1&1.2\\
1.3&1.4&1.5&1.6\\
1.7&1.8&1.9&2.0\\
2.1&2.2&2.3&2.4
\end{bmatrix}
$$

第 0 行对应 ID 0，第 1 行对应 ID 1，以此类推。这里的数值故意简单，真实 embedding 是训练得到的浮点参数。

现在分词器输出：

```text
文本：  hello world
Token： hello | world
ID：    2     | 5
```

向量化就是取 `E` 的第 2 行和第 5 行：

$$
X=E[[2,5]]=
\begin{bmatrix}
0.9&1.0&1.1&1.2\\
2.1&2.2&2.3&2.4
\end{bmatrix}
$$

输出 `X` 的形状是 `[2,4]`：2 个 token，每个 token 一个 4 维向量。

## 2. 为什么查表等价于 one-hot 乘矩阵

ID 2 可以表示成长度为 6 的 one-hot 向量：

$$
e_2=[0,0,1,0,0,0]
$$

如果把它当作行向量，乘以 `E`：

$$
e_2E=0E_0+0E_1+1E_2+0E_3+0E_4+0E_5=E_2
$$

所以 one-hot 乘 embedding 矩阵的结果就是对应行。查表实现只做这次索引，不创建长度为词表大小的 one-hot，节省时间和显存。

这也解释了为什么 ID 数值本身没有语义：one-hot 只用来选择行，ID 的大小不会进入点积或距离计算。

## 3. embedding 为什么是可学习的

初始化时，`E` 的每一行通常是小的随机数。训练时，语言模型根据预测损失反向传播，更新实际被使用的行。

例如输入 ID 2 参与了当前 batch 的预测，它对应的 embedding 行和后续层参数都会收到梯度。没有出现在 batch 里的行，通常不会从这一步得到梯度。

embedding 行不是分词器根据词义写出来的，也不是训练 tokenizer 时生成的数值。分词器决定“ID 2 是哪个 token”，模型训练决定“ID 2 对应的向量长什么样”。

## 4. 一个 token 如何进入第一层 Transformer

查表得到 `X_token` 之后，还要处理位置信息。不同模型的方式不同：

- 绝对位置 embedding：把 token 向量和位置向量相加；
- RoPE：先用 token 向量经过 Q/K 投影，再在 Attention 中旋转 Q/K；
- 其他相对位置方案：在 Attention 分数中加入位置偏置或采用其他变换。

不能把所有模型都写成 `embedding + position_embedding`。如果使用 RoPE，输入隐藏状态通常还是 token embedding，位置信息在后续 Q/K 上注入。

以绝对位置 embedding 为例，假设位置 0 的向量是 `p₀=[0.01,0.02,0.03,0.04]`，则：

$$
h_0=E_2+p_0=[0.91,1.02,1.13,1.24]
$$

这个相加要求两者形状相同。之后 `h₀` 会进入 Transformer block，而不是直接被当作 logits。

## 5. 从一个 token 到一个 batch：形状如何变化

假设 batch 中有 `B=2` 条样本，每条 padding 到 `L=3`，词表大小 `V=6`，隐藏维度 `D=4`：

```text
input_ids       [B, L]       = [2, 3]
embedding E     [V, D]       = [6, 4]
lookup 输出     [B, L, D]    = [2, 3, 4]
attention mask  [B, L]       = [2, 3]
```

`input_ids[b,t]` 选择 `E` 的第几行，得到 `hidden[b,t,:]`。padding 的 ID 也会查出一个向量，但 attention mask 和 loss mask 要确保它不被当成真实内容使用。

如果模型使用 tied embeddings（输入 embedding 与输出 softmax 权重共享），同一矩阵 `E` 还会参与从隐藏状态计算词表 logits：

$$
z=hE^T+b
$$

`h` 是 `[D]`，`Eᵀ` 是 `[D,V]`，所以 `z` 是 `[V]`，表示下一个 token 的每个候选 ID 的分数。这里的 logits 还不是概率，要再经过 Softmax 或采样策略。

## 6. 为什么模型输出的是 token ID，而不是直接输出文字

最后一层通常输出词表大小的 logits。模型选择下一个 token ID 后，分词器负责把 ID 解码为 token 字符串，再把字节或空白标记还原成文本。

所以生成链路是：

```text
文本 → tokenize → token IDs → embedding → Transformer
     ← decode  ← token IDs ← 采样 logits ← Transformer
```

模型内部主要在离散 ID 的 embedding 空间和隐藏空间里计算；“人看到的文字”只在输入编码和输出解码边界出现。

## 7. 向量化相关的面试追问

**为什么不用把 ID 除以词表大小得到一个数字？** 这样会给 ID 强行引入没有意义的顺序关系；embedding lookup 允许每个 token 学习独立向量。

**embedding lookup 和 one-hot 乘矩阵是否完全相同？** 数学结果相同，工程实现通常不显式创建 one-hot。

**为什么词表变了，原 embedding 不能直接复用？** 因为 ID 到 token 的映射可能改变，原矩阵每一行的语义对应关系也随之改变。

**padding 的 embedding 会不会产生梯度？** 取决于 loss mask、attention mask 和实现是否冻结 pad 行；不能只看 pad ID 的名字判断。

**RoPE 是不是加到 embedding 上？** 标准 RoPE 通常不直接加位置向量，而是在 Attention 中对 Q/K 做位置相关旋转。

下一步可以回到 [Attention 的缩放点积](../../architecture/attention/01-缩放点积注意力.md)，看这些 hidden states 如何生成 Q/K/V。
