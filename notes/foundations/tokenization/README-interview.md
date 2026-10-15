# Tokenization 面试提纲

面试准备不改变主线正文，因此单独列出答题路径：

1. 先定义 token、词表、ID、规范化和编码/解码。
2. 解释 BPE 的频次合并与 merge rank。
3. 解释 WordPiece 的典型最长匹配、`##` 和 `[UNK]`，并说明训练实现并不完整公开。
4. 解释 SentencePiece 是工具，可选 BPE 或 Unigram。
5. 解释 Unigram 的候选概率、动态规划和词表裁剪。
6. 补充 Unicode/UTF-8、byte fallback、padding/truncation/mask、特殊 token。
7. 最后谈 token 长度、embedding 成本、注意力 `L²` 成本、数据泄漏和复现配置。

对应详细内容见 [面试深度与常见陷阱](06-面试深度与常见陷阱.md)。
