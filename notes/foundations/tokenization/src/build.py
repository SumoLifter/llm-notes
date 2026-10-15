"""Generate formula-based illustrations and an offline reader for this topic."""
from pathlib import Path
import hashlib
import html
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, mathtext
from matplotlib.patches import FancyBboxPatch, Rectangle
import markdown
from PIL import Image, ImageOps

from demo import examples

TOPIC = Path(__file__).resolve().parents[1]
ASSETS = TOPIC / "assets"
BLUE, ORANGE, GREEN, PURPLE = "#24649a", "#ce7739", "#34857b", "#7962ac"
CHAPTERS = [
    ("01-从文本到Token.md", "chapter-01", "01 · 从文本到 token"),
    ("02-BPE与WordPiece.md", "chapter-02", "02 · BPE 与 WordPiece"),
    ("03-SentencePiece与Unigram.md", "chapter-03", "03 · SentencePiece / Unigram"),
    ("04-词表生成与工程实践.md", "chapter-04", "04 · 词表生成与实践"),
    ("05-实验与参考资料.md", "chapter-05", "05 · 实验与参考资料"),
    ("06-面试深度与常见陷阱.md", "chapter-06", "06 · 面试深度与陷阱"),
]


def configure():
    preferred = Path("C:/Windows/Fonts/msyh.ttc")
    if preferred.exists():
        font_manager.fontManager.addfont(str(preferred))
        family = font_manager.FontProperties(fname=str(preferred)).get_name()
    else:
        family = next((f.name for f in font_manager.fontManager.ttflist
                       if f.name in {"Noto Sans CJK SC", "SimHei"}), "DejaVu Sans")
    plt.rcParams.update({"font.family": family, "font.size": 12,
                         "axes.unicode_minus": False, "svg.fonttype": "path",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#c1c9d1", "text.color": "#263440",
                         "xtick.color": "#56616d", "ytick.color": "#56616d",
                         "figure.facecolor": "white", "savefig.dpi": 180})


def save(fig, name, title, note):
    fig.suptitle(title, x=0.055, y=0.97, ha="left", fontsize=18, fontweight="bold")
    fig.text(0.055, 0.035, note, color="#697582", fontsize=10)
    fig.subplots_adjust(left=0.09, right=0.96, bottom=0.19, top=0.79, wspace=0.38)
    for ext in ("png", "svg"):
        fig.savefig(ASSETS / f"{name}.{ext}")
    plt.close(fig)


def pipeline():
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.set(xlim=(0, 12), ylim=(0, 4))
    ax.axis("off")
    def box(x, y, label, color, width=1.8):
        ax.add_patch(FancyBboxPatch((x, y), width, 0.72,
                     boxstyle="round,pad=0.04,rounding_size=0.08", ec=color, fc="#f5f8fa", lw=1.4))
        ax.text(x + width / 2, y + 0.36, label, ha="center", va="center", fontsize=13)
    def arrow(x1, x2, y):
        ax.annotate("", (x2 - 0.08, y + 0.36), (x1 + 1.86, y + 0.36),
                    arrowprops={"arrowstyle": "->", "color": "#8999a7", "lw": 1.6})
    ax.text(0.05, 3.57, "训练：从语料生成词表", color=BLUE, fontsize=14, fontweight="bold")
    top = [0.15, 3.25, 6.35, 9.45]
    for x, label in zip(top, ["训练语料", "规范化与边界", "学习子词", "词表 + 规则"]):
        box(x, 2.45, label, BLUE)
    for x1, x2 in zip(top, top[1:]):
        arrow(x1, x2, 2.45)
    ax.text(0.05, 1.78, "编码：固定规则处理新文本", color=GREEN, fontsize=14, fontweight="bold")
    bottom = [0.15, 2.55, 4.95, 7.35, 9.75]
    for x, label in zip(bottom, ["新文本", "同样的预处理", "子词切分", "ID 序列", "模型"]):
        box(x, 0.65, label, GREEN)
    for x1, x2 in zip(bottom, bottom[1:]):
        arrow(x1, x2, 0.65)
    ax.annotate("", (5.85, 1.45), (10.35, 2.35),
                arrowprops={"arrowstyle": "->", "color": PURPLE, "lw": 1.5,
                            "connectionstyle": "arc3,rad=-0.08"})
    ax.text(8.3, 1.52, "编码使用已训练的规则", color=PURPLE, fontsize=11)
    save(fig, "01-tokenizer-pipeline", "词表训练与文本编码，是两个不同过程",
         "职责示意：具体预处理、文件格式与特殊 token 行为取决于工具配置。")


def bpe_merges(data):
    fig, ax = plt.subplots(figsize=(12, 5.6))
    ax.set(xlim=(0, 12), ylim=(0, 5))
    ax.axis("off")
    columns = [3.45, 5.15, 6.85, 8.55, 10.55]
    for x, (word, freq) in zip(columns, data["corpus"].items()):
        ax.text(x, 4.65, f"{word} × {freq}", ha="center", fontsize=13, fontweight="bold")
    for row, record in enumerate(data["bpe_history"][:4]):
        y = 3.85 - row * 1.02
        if row % 2 == 0:
            ax.add_patch(Rectangle((0, y - 0.36), 12, 0.78, fc="#f2f6f8", ec="none"))
        label = "初始字符" if row == 0 else f"{row}. 合并 {''.join(record['pair'])}（{record['count']} 次）"
        ax.text(0.15, y, label, va="center", fontsize=12, color=BLUE)
        for x, units in zip(columns, record["splits"].values()):
            ax.text(x, y, " | ".join(units), ha="center", va="center", fontsize=12)
    save(fig, "02-bpe-merges", "BPE：每轮根据更新后的相邻对计数继续合并",
         "教学语料：hug 10、pug 5、pun 12、bun 4、hugs 5；从字符开始，不跨词边界。")


def wordpiece_scores(data):
    candidates = data["wordpiece_teaching_history"][0]["candidates"]
    labels = [" + ".join(item["pair"]) for item in candidates]
    counts = [item["pair_count"] for item in candidates]
    scores = [item["score"] for item in candidates]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    positions = list(range(len(candidates)))
    for ax, values, kind in zip(axes, [counts, scores], ["相邻对加权计数", "教学归一化分数"]):
        winner = max(values)
        colors = [ORANGE if value == winner else BLUE for value in values]
        ax.barh(positions, values, color=colors, height=0.62)
        ax.set_yticks(positions, labels, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel(kind)
        ax.set_xlim(0, max(values) * 1.30)
        for y, value in zip(positions, values):
            ax.text(value + max(values) * 0.025, y,
                    str(value) if kind.startswith("相邻") else f"{value:.4f}",
                    va="center", fontsize=10)
    fig.subplots_adjust(wspace=0.65)
    save(fig, "03-wordpiece-scores", "出现得最多，与归一化分数最高，不是同一个标准",
         "同一初始语料；左图计数，右图 f(a,b)/(f(a)f(b))；不宣称复现原始 WordPiece 训练器。")


def unigram_paths(data):
    paths = data["unigram_paths"]
    fig, ax = plt.subplots(figsize=(11, 5.6))
    for y, record in enumerate(paths):
        pieces, p = record["pieces"], record["probability"]
        color = GREEN if pieces == ["ab", "ab"] else ORANGE if pieces == ["aba", "b"] else BLUE
        ax.barh(y, p, color=color, height=0.62)
        suffix = "  最优路径" if color == GREEN else "  最长匹配路径" if color == ORANGE else ""
        ax.text(p + 0.003, y, f"{p:.6f}".rstrip("0").rstrip(".") + suffix,
                va="center", fontsize=11)
    ax.set_yticks(range(len(paths)), [" | ".join(record["pieces"]) for record in paths])
    ax.invert_yaxis()
    ax.set(xlabel="整条切分路径的概率（片段概率相乘）", xlim=(0, 0.225))
    fig.subplots_adjust(left=0.20)
    save(fig, "04-unigram-paths", "Unigram：比较整条路径，不是逐步挑最长片段",
         "人工词表：ab=.40、a=.15、b=.15、aba=.10、bab=.20；输入 abab 的六条合法路径。")


def vocabulary_budget(data):
    history = data["bpe_history"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4))
    axes[0].plot([r["vocab_size"] for r in history], [r["token_count"] for r in history],
                 "o-", color=BLUE, lw=2.4)
    axes[0].set(xlabel="教学 BPE 词表项数", ylabel="加权语料 token 总数")
    axes[0].set_xticks([r["vocab_size"] for r in history])
    axes[0].grid(axis="y", alpha=0.17)
    vocab_sizes = [16000, 32000, 64000, 128000]
    params = [v * 4096 / 1e6 for v in vocab_sizes]
    axes[1].bar(range(4), params, color=[GREEN, GREEN, PURPLE, PURPLE], width=0.62)
    axes[1].set_xticks(range(4), ["16k", "32k", "64k", "128k"])
    axes[1].set(xlabel="假设词表大小 V", ylabel="单份 embedding 参数（百万）", ylim=(0, 620))
    for x, value in enumerate(params):
        axes[1].text(x, value + 12, f"{value:.1f}", ha="center", fontsize=11)
    save(fig, "05-vocabulary-budget", "词表变大：序列可能缩短，模型矩阵也会变大",
         "左：固定教学语料，7 次合并；右：仅算 V×4096 的单份 embedding，不是整个模型成本。")


def formula(match):
    expression = " ".join(match.group(1).split())
    digest = hashlib.sha256(expression.encode()).hexdigest()[:12]
    folder = ASSETS / "equations"
    folder.mkdir(exist_ok=True)
    with plt.rc_context({"savefig.transparent": True}):
        mathtext.math_to_image(f"${expression}$", folder / f"{digest}.svg",
                               prop=font_manager.FontProperties(size=18),
                               format="svg", color="#25313d")
    return (f'\n\n<div class="equation"><img src="assets/equations/{digest}.svg" '
            f'alt="{html.escape(expression, quote=True)}"></div>\n\n')


def reader():
    css = (TOPIC / "src/reader.css").read_text(encoding="utf-8")
    sections = []
    for filename, anchor, _ in CHAPTERS:
        text = (TOPIC / filename).read_text(encoding="utf-8")
        text = re.sub(r"\$\$\s*\n(.*?)\n\s*\$\$", formula, text, flags=re.S)
        for source, target, _ in CHAPTERS:
            text = text.replace(f"]({source})", f"](#{target})")
        body = markdown.markdown(text, extensions=["tables", "fenced_code", "toc"])
        def expandable_figure(match):
            tag = match.group(1)
            source = re.search(r'src="([^"]+)"', tag).group(1)
            if not source.endswith(".png"):
                return match.group(0)
            vector = source[:-4] + ".svg"
            return (f'<figure class="illustration"><a href="{vector}" target="_blank" '
                    f'rel="noopener">{tag}</a><figcaption>点击插图查看可缩放的 SVG 大图</figcaption></figure>')
        body = re.sub(r"<p>(<img [^>]+>)</p>", expandable_figure, body)
        body = re.sub(r'id="([^"]+)"', lambda m: f'id="{anchor}-{m.group(1)}"', body)
        sections.append(f'<section class="chapter" id="{anchor}">{body}</section>')
    links = "".join(f'<a href="#{anchor}">{label}</a>' for _, anchor, label in CHAPTERS)
    document = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tokenization 算法与词表生成机制 · 阅读版</title><style>{css}</style></head>
<body><nav aria-label="章节目录"><div class="brand">大模型 · 基础原理</div>{links}
<hr><a href="README.md">专题源文件</a><a href="../../../README.md">资料总入口</a>
<p>先分清词表训练和文本编码，再比较具体方法。<br>图、公式均可离线显示。</p></nav>
<main><article><header><div class="eyebrow">从文本到模型输入</div>
<h1>Tokenization 算法<br>与词表生成机制</h1>
<p>词表是怎么学出来的？新词又怎样切分？从一份可以手算的小语料开始，理解 BPE、WordPiece 与 SentencePiece。</p>
<p>SentencePiece 是工具，可使用 BPE 或 Unigram。示例结果来自教学代码，不代表任何现成模型的词表。</p></header>
{''.join(sections)}<footer>Markdown 为正文源文件，Python 生成插图与阅读版。编写日期：2026-10-14。</footer>
</article></main></body></html>'''
    (TOPIC / "阅读版.html").write_text(document, encoding="utf-8")


def overview():
    sheet = Image.new("RGB", (1600, 1200), "#e9edf0")
    for i, path in enumerate(sorted(ASSETS.glob("[0-9][0-9]-*.png"))):
        with Image.open(path) as source:
            image = ImageOps.contain(source.convert("RGB"), (780, 380))
            sheet.paste(image, ((i % 2) * 800 + (800 - image.width) // 2,
                               (i // 2) * 400 + 10))
    sheet.save(ASSETS / "overview.png")


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    configure()
    data = examples()
    pipeline()
    for draw in (bpe_merges, wordpiece_scores, unigram_paths, vocabulary_budget):
        draw(data)
    (TOPIC / "src/example-results.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    reader()
    overview()
    print("Built 5 figures (PNG + SVG), formula SVGs, example results, and HTML reader.")


if __name__ == "__main__":
    main()
