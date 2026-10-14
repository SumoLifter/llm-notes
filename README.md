# 大模型学习与研究资料

从这里进入各个专题。知识笔记、论文原文和实验记录分开存放，避免以后所有资料都挤在根目录。

## 当前专题

- [旋转位置编码与上下文长度扩展](notes/foundations/rope-and-context/README.md)：从注意力基础讲到 RoPE、PI、NTK-aware、YaRN，以及扩展训练与评测。
- [打开舒适阅读版](notes/foundations/rope-and-context/阅读版.html)：本地 HTML，调整了行距、段间距、公式和插图布局，无需联网。

## 资料放在哪里

```text
大模型/
├── README.md                 资料入口与存放规则
├── notes/                    自己整理的知识笔记
│   └── foundations/          基础原理：注意力、位置编码等
│       └── rope-and-context/ 当前专题：正文、图和绘图源码
├── papers/                   论文 PDF、补充材料与论文阅读笔记
├── experiments/              实验代码、配置、结果与运行记录
├── resources/                跨专题复用的资料、模板与数据说明
├── scripts/                  跨专题使用的工具脚本
└── archive/                  原稿、旧版本与不再维护的材料
```

`notes` 后续按内容增加 `training`、`inference`、`evaluation`、`applications` 等目录。有实际资料时再创建，不提前堆空目录。

## 新资料的组织规则

**知识笔记以专题为单位。** 一个专题放一个文件夹，入口叫 `README.md`。正文超过几个大节时拆成带编号的章节，例如 `01-基础.md`。图片放在专题自己的 `assets/`，生成它们的代码放在 `src/`。

**论文以论文为单位。** 推荐 `papers/年份-短标题/`，里面放原文、来源链接和阅读笔记。笔记引用论文即可，不把同一个 PDF 复制到多个专题。

**实验以实验为单位。** 推荐 `experiments/日期-实验名称/`，至少记录目的、模型版本、配置、运行方法和结论。大型权重、数据集和缓存放在专门的数据盘位置，这里只记录路径与获取方法。

**原稿只归档，不继续修改。** 当前原稿保存在 [archive/originals](archive/originals/README.md)。根目录旧文件名保留为跳转入口，方便之前的链接继续使用。

## 重建当前专题

```powershell
python notes/foundations/rope-and-context/src/build.py
```

该命令重绘所有插图，并由 Markdown 生成 HTML 阅读版。依赖见专题内的 `src/requirements.txt`。
