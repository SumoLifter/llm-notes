# Tokenization 算法与词表生成机制

模型不能直接接收一段文字。它先把文字转换成 token，再把 token 转成整数编号。分词器决定了这一步如何进行，也决定同一段内容要占用多少上下文位置。

本专题回答两个不同的问题：**词表是怎样从语料里生成的？词表固定后，新文本又怎样切分？**

## 阅读顺序

1. [从文本到 token ID](01-从文本到Token.md)：字符、字节、子词、词表，以及完整处理流程。
2. [BPE 与 WordPiece](02-BPE与WordPiece.md)：逐步计算训练例子，比较推理时的切分规则。
3. [SentencePiece 与 Unigram](03-SentencePiece与Unigram.md)：工具与算法的区别，概率切分与词表裁剪。
4. [词表生成与工程实践](04-词表生成与工程实践.md)：语料、覆盖率、特殊 token、词表规模与模型兼容性。
5. [可复现实验与参考资料](05-实验与参考资料.md)：运行教学实现，核对原始资料与解释边界。
6. [面试深度与常见陷阱](06-面试深度与常见陷阱.md)：复杂度、Unicode/字节边界、训练与编码分离、工程排错和高频面试追问。
7. [Token 向量化的具体计算](07-Token向量化的具体计算.md)：从 token ID 查表，到位置处理、batch 形状和进入 Transformer 的数值过程。

[舒适阅读版](阅读版.html) 将章节连成一篇，保留较大的字号、行距和图文留白，公式本地渲染，无需联网。

## 约定与文件

文中的词表、ID、概率和训练语料都是明确指定的教学例子，不是任何现成模型的真实分词结果。WordPiece 的归一化合并分数是常见教学近似，不宣称还原原始闭源训练算法。

- `assets/`：Python 生成的 PNG/SVG 插图与公式。
- `src/demo.py`：BPE、教学版 WordPiece 训练、贪心 WordPiece 切分和 Unigram 路径计算。
- `src/build.py`：运行示例、重绘插图、生成 HTML 与实验记录。
- `src/example-results.json`：可重建的例子输出，便于核对正文数字。
- [面试提纲](README-interview.md)：适合面试前快速复习的答题路径。

后续修改请遵循 [资料库写作与阅读偏好](../../../README.md#写作与阅读偏好后续修改请遵循)。图和公式附近解释概念，不把实现名称与算法名称混为一谈。

## 重建

在资料库根目录执行：

```powershell
python notes/foundations/tokenization/src/demo.py
python notes/foundations/tokenization/src/build.py
```

`demo.py` 只依赖 Python 标准库。阅读版与绘图依赖见 `src/requirements.txt`。当前教学实现不替代生产分词器。

返回 [资料总入口](../../../README.md)。
