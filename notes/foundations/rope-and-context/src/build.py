"""Rebuild teaching figures and an offline HTML reader from this topic's Markdown.

Run from any directory: python path/to/rope-and-context/src/build.py
Figures are mathematical illustrations, not measurements of a trained model.
"""
from __future__ import annotations

import hashlib
import html
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, mathtext
from matplotlib.patches import Arc
import markdown
import numpy as np
from PIL import Image, ImageOps, ImageDraw

TOPIC = Path(__file__).resolve().parents[1]
ASSETS = TOPIC / "assets"
D, BASE, TRAIN, TARGET = 128, 10_000, 4096, 32768
ALPHA = TARGET / TRAIN
BLUE, ORANGE, GREEN, PURPLE = "#24649a", "#ce7739", "#34857b", "#7962ac"
CHAPTERS = [
    ("01-RoPE基础.md", "chapter-01", "01 · RoPE 基础"),
    ("02-长度扩展方法.md", "chapter-02", "02 · 长度扩展方法"),
    ("03-扩展训练实践.md", "chapter-03", "03 · 扩展训练实践"),
    ("04-数学补充.md", "chapter-04", "04 · 数学补充"),
    ("参考资料.md", "references", "参考资料"),
]


def configure_style():
    preferred = Path("C:/Windows/Fonts/msyh.ttc")
    if preferred.exists():
        font_manager.fontManager.addfont(str(preferred))
        font = font_manager.FontProperties(fname=str(preferred)).get_name()
    else:
        font = next((f.name for f in font_manager.fontManager.ttflist
                     if f.name in {"Noto Sans CJK SC", "Microsoft YaHei", "SimHei"}),
                    "DejaVu Sans")
    plt.rcParams.update({
        "font.family": font, "font.size": 11,
        "axes.unicode_minus": False, "axes.spines.top": False,
        "axes.spines.right": False, "axes.labelcolor": "#46505b",
        "text.color": "#25313d", "axes.edgecolor": "#c1c9d1",
        "xtick.color": "#56616d", "ytick.color": "#56616d",
        "figure.facecolor": "#ffffff", "axes.facecolor": "#ffffff",
        "svg.fonttype": "path", "savefig.dpi": 180,
    })


def frequencies(base=BASE):
    return base ** (-2 * np.arange(D // 2) / D)


def finish(fig, name, title, note):
    fig.suptitle(title, x=0.06, y=0.98, ha="left", fontsize=18, fontweight="bold")
    fig.text(0.06, 0.025, note, fontsize=10, color="#697582")
    fig.subplots_adjust(left=0.085, right=0.965, bottom=0.19, top=0.80, wspace=0.32)
    for ext in ("png", "svg"):
        fig.savefig(ASSETS / f"{name}.{ext}")
    plt.close(fig)


def rotation_figure():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, m, n in zip(axes, (7, 17), (12, 22)):
        ax.add_patch(plt.Circle((0, 0), 1, fill=False, color="#c8d0d7", ls="--"))
        ax.axhline(0, color="#e3e8ed", lw=1)
        ax.axvline(0, color="#e3e8ed", lw=1)
        angles = (30 + m * 10, 10 + n * 10)
        for angle, color, label in zip(angles, (BLUE, ORANGE), ("Q", "K")):
            rad = np.deg2rad(angle)
            x, y = np.cos(rad), np.sin(rad)
            ax.annotate("", (x, y), (0, 0),
                        arrowprops={"arrowstyle": "-|>", "color": color, "lw": 2.8})
            ax.text(x * 1.15, y * 1.15, f"{label} {angle % 360}°", color=color,
                    ha="center", va="center", fontsize=12)
        ax.add_patch(Arc((0, 0), 0.72, 0.72, theta1=angles[0], theta2=angles[1],
                         color=GREEN, lw=2))
        mid = np.deg2rad(np.mean(angles))
        ax.text(0.53 * np.cos(mid), 0.53 * np.sin(mid), "30°", color=GREEN,
                ha="center", va="center")
        ax.set(xlim=(-1.42, 1.42), ylim=(-1.42, 1.42), aspect="equal")
        ax.set_title(f"位置 {m} 与 {n} · 相对位置差不变", pad=14)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    finish(fig, "01-relative-rotation", "共同平移位置，夹角保持不变",
           "教学例：每 token 旋转 10°；旋转前 Q 为 30°、K 为 10°；向量长度均为 1。")


def frequency_figure():
    omega = frequencies()
    wave = 2 * np.pi / omega
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    axes[0].semilogy(np.arange(64), wave, color=BLUE, lw=2.4)
    for i, color in zip((0, 32, 63), (ORANGE, GREEN, PURPLE)):
        axes[0].scatter(i, wave[i], color=color, s=40, zorder=3)
    axes[0].set(xlabel="坐标对编号 i（越右越慢）", ylabel="转一整圈所需 token 数")
    axes[0].grid(axis="y", alpha=0.18)
    positions = np.arange(201)
    for i, color in zip((0, 32, 63), (ORANGE, GREEN, PURPLE)):
        axes[1].plot(positions, np.cos(positions * omega[i]), color=color,
                     lw=1.7, label=f"i={i}，波长≈{wave[i]:,.0f}")
    axes[1].set(xlabel="位置 m", ylabel=r"$\cos(m\omega_i)$", ylim=(-1.15, 1.15))
    axes[1].legend(loc="lower right", fontsize=9, frameon=False)
    finish(fig, "02-frequency-scales", "快慢不同的频率，提供不同尺度的位置特征",
           "d=128，b=10,000；右图取整数位置采样，没有把单个通道当作模型能力。")


def phase_figure():
    omega = frequencies()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for length, color, label in ((TRAIN, BLUE, "4K 训练长度"),
                                  (TARGET, ORANGE, "32K 目标长度")):
        axes[0].semilogy(np.arange(64), length * omega / (2 * np.pi),
                         color=color, lw=2.4, label=label)
    axes[0].axhline(1, color="#9da8b3", ls="--", lw=1)
    axes[0].set(xlabel="坐标对编号 i", ylabel="未取模的旋转圈数")
    axes[0].legend(frameon=False, fontsize=10)
    lengths = np.arange(TARGET + 1)
    axes[1].axvspan(0, TRAIN, color=BLUE, alpha=0.08)
    axes[1].plot(lengths, lengths * omega[-1] / (2 * np.pi), color=PURPLE, lw=2.4)
    for length, color in ((TRAIN, BLUE), (TARGET, ORANGE)):
        value = length * omega[-1] / (2 * np.pi)
        axes[1].scatter(length, value, color=color, s=45, zorder=3)
        axes[1].annotate(f"{value:.3f} 圈", (length, value),
                         xytext=(0 if length == TRAIN else -8, 12),
                         textcoords="offset points", ha="left" if length == TRAIN else "right")
    axes[1].set(xlabel="序列长度边界", ylabel="最慢通道 i=63 的圈数", ylim=(0, 0.72))
    axes[1].set_xticks([0, TRAIN, 16384, TARGET], ["0", "4K", "16K", "32K"])
    finish(fig, "03-phase-coverage", "较慢的通道会进入原长度未覆盖的角度范围",
           "d=128，b=10,000；横轴使用长度边界说明尺度，实际位置从 0 开始。")


def yarn_scale():
    # Original-style correction boundaries, then a linear ramp in pair index.
    fast = np.floor(D * np.log(TRAIN / (32 * 2 * np.pi)) / (2 * np.log(BASE)))
    slow = np.ceil(D * np.log(TRAIN / (1 * 2 * np.pi)) / (2 * np.log(BASE)))
    indices = np.arange(D // 2)
    gamma = 1 - np.clip((indices - fast) / (slow - fast), 0, 1)
    return gamma + (1 - gamma) / ALPHA, int(fast), int(slow)


def scaling_figure():
    i = np.arange(D // 2)
    yarn, fast, slow = yarn_scale()
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.axvspan(0, fast, color=BLUE, alpha=0.06)
    ax.axvspan(slow, 63, color=PURPLE, alpha=0.06)
    ax.plot(i, np.full(64, 1 / ALPHA), color=ORANGE, lw=2.5, ls="--", label="PI")
    ax.plot(i, ALPHA ** (-2 * i / (D - 2)), color=GREEN, lw=2.5, label="静态 NTK-aware")
    ax.plot(i, yarn, color=PURPLE, lw=2.8, label="YaRN（通道线性 ramp）")
    ax.set(xlabel="坐标对编号 i（越右频率越低）", ylabel="新频率 / 原频率", xlim=(0, 63), ylim=(0, 1.12))
    ax.set_yticks([0.125, 0.25, 0.5, 0.75, 1], ["1/8", "1/4", "1/2", "3/4", "1"])
    ax.legend(frameon=False, fontsize=11, loc="upper right")
    ax.grid(axis="y", alpha=0.18)
    finish(fig, "04-scaling-methods", "三种方法如何分配频率缩放",
           f"d=128，b=10,000，4K→32K；YaRN βfast=32、βslow=1；过渡编号 {fast}—{slow}。")


def attention_figure():
    # One relevant score A=8, all N-1 distractor scores exactly zero.
    A = 8.0
    c = 1 + 0.1 * np.log(ALPHA)
    temp = 1 / c ** 2
    counts = np.array([1024, 4096, 16384, 32768, 131072])
    probability = lambda n, t: 1 / (1 + (n - 1) * np.exp(-A / t))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for t, color, label in ((1, BLUE, "原尺度 t=1"), (temp, ORANGE, f"示例尺度 t≈{temp:.2f}")):
        axes[0].plot(counts, probability(counts, t) * 100, "o-", color=color, lw=2.4, label=label)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(counts, ["1K", "4K", "16K", "32K", "128K"])
    axes[0].set(xlabel="候选位置数量 N", ylabel="相关位置的权重（%）", ylim=(0, 105))
    axes[0].legend(frameon=False, fontsize=9)
    temps = np.linspace(0.5, 1.3, 200)
    axes[1].plot(temps, probability(TARGET, temps) * 100, color=PURPLE, lw=2.4)
    for t, color in ((1, BLUE), (temp, ORANGE)):
        p = probability(TARGET, t) * 100
        axes[1].scatter(t, p, color=color, s=40)
        axes[1].annotate(f"t={t:.2f}：{p:.1f}%", (t, p), xytext=(10, 10),
                         textcoords="offset points", fontsize=10)
    axes[1].set(xlabel="温度 t（越小，分数差越大）", ylabel="相关位置的权重（%）", ylim=(0, 105))
    finish(fig, "05-attention-scaling", "分数差固定时，候选数量与温度会影响权重",
           "人工模拟：一个相关分数 A=8，其余为 0；右图 N=32,768；不是实际模型测试。")


def kernel_figure():
    distance = np.arange(2001)
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for base, color in ((1000, "#87939f"), (10000, BLUE), (100000, PURPLE)):
        kernel = np.cos(distance[:, None] * frequencies(base)[None, :]).mean(axis=1)
        ax.plot(distance, kernel, color=color, lw=1.2, alpha=0.88, label=f"b={base:,}")
    ax.axhline(0, color="#bdc6cf", lw=1)
    ax.set(xlabel="相对距离 Δ（token）", ylabel="等权余弦平均 C(Δ)", ylim=(-0.4, 1.1))
    ax.legend(frameon=False, fontsize=11)
    finish(fig, "06-position-kernel", "频率叠加会振荡：不能当作单调衰减的模型分数",
           "d=128，各通道等权；显示全部整数距离，无平滑；不是注意力权重或远程信息量。")


def render_equation(match):
    expression = " ".join(match.group(1).split())
    # Mathtext supports sized delimiters via left/right, not LaTeX's big command.
    rendered = expression.replace(r"\big(", r"\left(").replace(r"\big)", r"\right)")
    digest = hashlib.sha256(expression.encode()).hexdigest()[:12]
    folder = ASSETS / "equations"
    folder.mkdir(exist_ok=True)
    target = folder / f"{digest}.svg"
    with plt.rc_context({"savefig.transparent": True}):
        mathtext.math_to_image(f"${rendered}$", target,
                               prop=font_manager.FontProperties(size=18),
                               format="svg", color="#25313d")
    return (f'\n\n<div class="equation"><img src="assets/equations/{digest}.svg" '
            f'alt="{html.escape(expression, quote=True)}"></div>\n\n')


CSS = """
:root{--ink:#263440;--muted:#72808b;--accent:#24649a;--paper:#fff;--bg:#f5f4ef}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:36px}
body{margin:0;background:var(--bg);color:var(--ink);font-family:'Microsoft YaHei','Noto Sans CJK SC',sans-serif}
nav{position:fixed;top:0;left:0;width:260px;height:100vh;padding:48px 28px;overflow:auto;border-right:1px solid #e1e4e5}
nav .brand{font-size:12px;letter-spacing:.15em;color:var(--muted);margin-bottom:32px}
nav a{display:block;font-size:14px;line-height:1.8;padding:10px 0;color:#4c6170;text-decoration:none}
nav a:hover{color:var(--accent)}nav hr{border:0;border-top:1px solid #dfe4e7;margin:28px 0}
nav p{font-size:12px;line-height:1.9;color:var(--muted)}
main{margin:0 0 0 260px;padding:64px 56px 120px}article{max-width:810px;margin:auto}
header{padding:24px 0 64px}.eyebrow{font-size:13px;color:var(--accent);letter-spacing:.12em}
header h1{font-size:36px;line-height:1.55;margin:20px 0 24px;letter-spacing:.01em}
header p{color:#617381;font-size:19px;line-height:1.9;max-width:660px}
.chapter{background:var(--paper);padding:52px 52px 60px;margin:0 0 72px;border:1px solid #e6e9eb;border-radius:8px}
.chapter h1{font-size:28px;line-height:1.65;margin:0 0 32px;font-weight:650}
.chapter h2{font-size:23px;line-height:1.65;margin:64px 0 24px;padding-top:12px;border-top:1px solid #e7ecef;font-weight:650}
.chapter h3{font-size:20px;line-height:1.7;margin:36px 0 16px;font-weight:650}
p,li{font-size:19px;line-height:1.95;overflow-wrap:break-word}
p{margin:0 0 26px}ul,ol{padding-left:1.5em;margin:24px 0 32px}li{margin:0 0 14px}
strong{font-weight:650;color:#233e50}a{color:var(--accent);text-underline-offset:4px}
code{font-family:'Microsoft YaHei',sans-serif;font-size:.91em;background:#f0f4f7;border-radius:4px;padding:2px 5px}
pre{overflow-x:auto;background:#f7f9fa;padding:24px 18px;margin:28px 0;border-radius:6px;line-height:1.9}pre code{font-family:Consolas,'Courier New',monospace;background:transparent;padding:0;font-size:15px}
.equation{margin:32px 0 36px;padding:28px 20px;text-align:center;background:#f7f9fa;border-radius:6px;overflow-x:auto}
.equation img{max-width:100%;height:auto;vertical-align:middle}
p:has(>img){margin:38px -24px 28px;padding:12px 0;background:white}
p>img{display:block;width:100%;height:auto}
table{border-collapse:collapse;width:100%;margin:32px 0;font-size:15px;line-height:1.8}
th,td{text-align:left;padding:14px 12px;border-bottom:1px solid #e1e7eb;vertical-align:top}
th{background:#f1f5f7;font-weight:650}blockquote{border-left:3px solid #b6cbd8;margin:28px 0;padding-left:20px}
footer{font-size:13px;line-height:1.9;color:var(--muted)}
@media(min-width:1450px){main{padding-right:130px}}
@media(max-width:1050px){nav{width:210px;padding:32px 20px}main{margin-left:210px;padding:40px 24px}.chapter{padding:38px 32px}p,li{font-size:18px}}
@media(max-width:760px){nav{position:static;width:auto;height:auto;padding:24px;border:0;border-bottom:1px solid #ddd}nav .brand{margin-bottom:12px}nav a{display:inline-block;margin-right:14px;font-size:13px}nav hr,nav p{display:none}main{margin:0;padding:28px 16px 60px}header{padding:10px 8px 32px}header h1{font-size:28px}.chapter{padding:30px 22px;margin-bottom:36px}.chapter h1{font-size:24px}.chapter h2{font-size:21px;margin-top:48px}p,li{font-size:17px;line-height:1.9}p:has(>img){margin-left:-12px;margin-right:-12px}table{font-size:13px}th,td{padding:10px 6px}.equation{padding:22px 10px}}
@media print{nav{display:none}main{margin:0;padding:0}body{background:white}.chapter{border:0;padding:24px 0;break-before:page}h1,h2,h3{break-after:avoid}.equation,p:has(>img){break-inside:avoid}header{padding-bottom:30px}}
"""


def build_reader():
    linkmap = {name: "#" + anchor for name, anchor, _ in CHAPTERS}
    sections = []
    for name, anchor, _ in CHAPTERS:
        text = (TOPIC / name).read_text(encoding="utf-8")
        text = re.sub(r"\$\$\s*\n(.*?)\n\s*\$\$", render_equation, text, flags=re.S)
        for source, dest in linkmap.items():
            text = text.replace(f"]({source})", f"]({dest})")
        converter = markdown.Markdown(extensions=["tables", "fenced_code", "toc"],
                                       extension_configs={"toc": {"toc_depth": "2-3"}})
        body = converter.convert(text)
        # Prefix heading anchors to avoid duplicated section IDs across chapters.
        body = re.sub(r'id="([^"]+)"', lambda m: f'id="{anchor}-{m.group(1)}"', body)
        sections.append(f'<section class="chapter" id="{anchor}">{body}</section>')
    links = "".join(f'<a href="#{anchor}">{label}</a>' for _, anchor, label in CHAPTERS)
    document = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>旋转位置编码与上下文长度扩展 · 阅读版</title><style>{CSS}</style></head>
<body><nav aria-label="章节目录"><div class="brand">大模型 · 基础原理</div>{links}
<hr><a href="README.md">专题源文件</a><a href="../../../README.md">资料总入口</a>
<p>先读前三章。数学补充可以按需查阅。<br>图与公式均可离线显示。</p></nav>
<main><article><header><div class="eyebrow">从原理到实践</div><h1>旋转位置编码<br>与上下文长度扩展</h1>
<p>先弄清楚位置如何进入注意力，再看频率如何调整，最后讨论训练和评测。</p>
<p>示例使用 128 维 RoPE，从 4K 扩展到 32K。token 是分词单元，不等于汉字或单词。</p></header>
{''.join(sections)}<footer>Markdown 是正文源文件，HTML 与插图由 Python 构建。修订日期：2026-10-14。</footer>
</article></main></body></html>'''
    (TOPIC / "阅读版.html").write_text(document, encoding="utf-8")


def contact_sheet():
    thumbs = []
    for path in sorted(ASSETS.glob("[0-9][0-9]-*.png")):
        with Image.open(path) as image:
            thumb = ImageOps.contain(image.convert("RGB"), (770, 370))
            tile = Image.new("RGB", (800, 400), "#e9edf0")
            tile.paste(thumb, ((800 - thumb.width) // 2, 10))
            ImageDraw.Draw(tile).text((16, 382), path.stem, fill="#30485b")
            thumbs.append(tile)
    sheet = Image.new("RGB", (1600, 1200), "white")
    for i, tile in enumerate(thumbs):
        sheet.paste(tile, ((i % 2) * 800, (i // 2) * 400))
    sheet.save(ASSETS / "overview.png")


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    configure_style()
    for draw in (rotation_figure, frequency_figure, phase_figure,
                 scaling_figure, attention_figure, kernel_figure):
        draw()
    build_reader()
    contact_sheet()
    print(f"Built 6 figures (PNG + SVG), equations, and {TOPIC / '阅读版.html'}")


if __name__ == "__main__":
    main()
