"""Build attention teaching figures and an offline HTML reader."""
from pathlib import Path
import hashlib
import html
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, mathtext
from matplotlib.patches import FancyBboxPatch, Rectangle
import markdown
from PIL import Image, ImageOps

TOPIC = Path(__file__).resolve().parents[1]
ASSETS = TOPIC / "assets"
BLUE, ORANGE, GREEN, PURPLE = "#24649a", "#ce7739", "#34857b", "#7962ac"
CHAPTERS = [
    ("01-缩放点积注意力.md", "chapter-01", "01 · 缩放点积注意力"),
    ("02-多头注意力.md", "chapter-02", "02 · 多头注意力"),
    ("03-MHA-MQA-GQA.md", "chapter-03", "03 · MHA / MQA / GQA"),
    ("04-训练与推理实现.md", "chapter-04", "04 · 训练与推理实现"),
    ("05-面试题与推导.md", "chapter-05", "05 · 面试题与推导"),
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
    fig.subplots_adjust(left=0.09, right=0.96, bottom=0.19, top=0.79, wspace=0.35)
    for ext in ("png", "svg"):
        fig.savefig(ASSETS / f"{name}.{ext}")
    plt.close(fig)


def flow():
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.set(xlim=(0, 12), ylim=(0, 4)); ax.axis("off")
    def box(x, y, label, color, width=1.55):
        ax.add_patch(FancyBboxPatch((x, y), width, .70, boxstyle="round,pad=.04,rounding_size=.08", ec=color, fc="#f5f8fa", lw=1.4))
        ax.text(x+width/2, y+.35, label, ha="center", va="center", fontsize=12)
    def arrow(x1, x2, y):
        ax.annotate("", (x2-.08,y+.35), (x1+1.63,y+.35), arrowprops={"arrowstyle":"->","color":"#8999a7","lw":1.6})
    ax.text(.1,3.45,"输入 X",color=BLUE,fontsize=14,fontweight="bold")
    top=[.3,2.3,4.3,6.3,8.3]
    for x,label in zip(top,["X","XW_Q → Q","XW_K → K","XW_V → V","QKᵀ / √dₖ"]): box(x,2.45,label,BLUE)
    for a,b in zip(top,top[1:]): arrow(a,b,2.45)
    bottom=[4.0,6.2,8.4]
    for x,label in zip(bottom,["mask + Softmax","注意力权重 A","A V → 输出"]): box(x,.7,label,GREEN,1.75)
    for a,b in zip(bottom,bottom[1:]):
        ax.annotate("",(b-.08,1.05),(a+1.83,1.05),arrowprops={"arrowstyle":"->","color":"#8999a7","lw":1.6})
    ax.annotate("",(4.85,1.45),(7.05,2.38),arrowprops={"arrowstyle":"->","color":PURPLE,"lw":1.5,"connectionstyle":"arc3,rad=.1"})
    ax.annotate("",(6.75,1.45),(5.05,2.38),arrowprops={"arrowstyle":"->","color":PURPLE,"lw":1.5,"connectionstyle":"arc3,rad=-.1"})
    ax.text(4.92,1.7,"分数矩阵",color=PURPLE,fontsize=11)
    save(fig,"01-attention-flow","Attention 的数据流：先算匹配，再汇总内容","示意形状：Q/K/V 投影 → 分数 → mask/Softmax → 用权重汇总 V。")


def head_sharing():
    fig, axes = plt.subplots(3,1,figsize=(11,6.2))
    configs=[("MHA",8,8,BLUE),("GQA",8,2,GREEN),("MQA",8,1,ORANGE)]
    for ax,(name,hq,hkv,color) in zip(axes,configs):
        ax.set(xlim=(-.5,8.5),ylim=(-.2,1.2)); ax.axis("off")
        ax.text(-.35,.5,name,ha="right",va="center",fontsize=15,fontweight="bold",color=color)
        for q in range(hq):
            ax.add_patch(Rectangle((q,.68),.68,.35,fc="#eaf2f6",ec=BLUE,lw=1))
            ax.text(q+.34,.855,f"Q{q}",ha="center",va="center",fontsize=10)
            kv=min(q*hkv//hq,hkv-1)
            x=kv*(8/hkv)/1.0 + (8/hkv-.68)/2
            ax.add_patch(Rectangle((x,.08),.68,.35,fc="#eef6f2",ec=GREEN,lw=1))
            ax.text(x+.34,.255,f"KV{kv}",ha="center",va="center",fontsize=9)
            ax.annotate("",(x+.34,.47),(q+.34,.66),arrowprops={"arrowstyle":"-","color":color,"lw":1.1,"alpha":.75})
        ax.text(7.75,.86,f"h_q={hq}",fontsize=10); ax.text(7.75,.22,f"h_kv={hkv}",fontsize=10)
    save(fig,"02-head-sharing","MHA、GQA、MQA 的头映射","示例 h_q=8；GQA 每 4 个 Q 头共享一个 KV 头；MQA 共享一组 KV。")


def cache():
    fig, ax=plt.subplots(figsize=(10,5.2))
    names=["MHA\nh_kv=8","GQA\nh_kv=2","MQA\nh_kv=1"]
    vals=[8,2,1]
    bars=ax.bar(names,vals,color=[BLUE,GREEN,ORANGE],width=.58)
    ax.set(ylabel="每层、每 batch、每位置的 KV 元素相对量",ylim=(0,8.8))
    ax.grid(axis="y",alpha=.18)
    for bar,val in zip(bars,vals): ax.text(bar.get_x()+bar.get_width()/2,val+.25,f"{val}/8",ha="center",fontsize=12)
    save(fig,"03-kv-cache","共享 KV 头，直接改变缓存规模","比例示意：固定 h_q=8 与 d_h；实际显存还受 batch、长度、层数、dtype 和布局影响。")


def formula(match):
    expression=" ".join(match.group(1).split()); digest=hashlib.sha256(expression.encode()).hexdigest()[:12]
    rendered=expression
    rendered=rendered.replace(r"\bmod", "mod").replace(r"\qquad", r"\quad")
    if "\\begin{cases}" in rendered:
        rendered=(r"S'_{ij}=S_{ij}\ (j\leq i);\quad "
                  r"S'_{ij}=-\infty\ (j>i)")
    folder=ASSETS/"equations"; folder.mkdir(exist_ok=True)
    with plt.rc_context({"savefig.transparent":True}):
        mathtext.math_to_image(f"${rendered}$",folder/f"{digest}.svg",prop=font_manager.FontProperties(size=18),format="svg",color="#25313d")
    return f'\n\n<div class="equation"><img src="assets/equations/{digest}.svg" alt="{html.escape(expression,quote=True)}"></div>\n\n'


def reader(css):
    sections=[]
    for filename,anchor,_ in CHAPTERS:
        text=(TOPIC/filename).read_text(encoding="utf-8")
        text=re.sub(r"\$\$\s*\n(.*?)\n\s*\$\$",formula,text,flags=re.S)
        for source,target,_ in CHAPTERS: text=text.replace(f"]({source})",f"](#{target})")
        body=markdown.markdown(text,extensions=["tables","fenced_code","toc"])
        body=re.sub(r'id="([^"]+)"',lambda m:f'id="{anchor}-{m.group(1)}"',body)
        body=re.sub(r'<p>(<img [^>]+>)</p>',lambda m:f'<figure class="illustration"><a href="{re.search(r"src=\"([^\"]+)",m.group(1)).group(1)[:-4]}.svg" target="_blank">{m.group(1)}</a><figcaption>点击插图查看可缩放的 SVG 大图</figcaption></figure>',body)
        sections.append(f'<section class="chapter" id="{anchor}">{body}</section>')
    links="".join(f'<a href="#{a}">{label}</a>' for _,a,label in CHAPTERS)
    document=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Attention 与 MHA/MQA/GQA · 阅读版</title><style>{css}</style></head><body><nav aria-label="章节目录"><div class="brand">大模型 · 模型架构</div>{links}<hr><a href="README.md">专题源文件</a><a href="../../../README.md">资料总入口</a><p>先看公式和形状，再看 KV cache 与工程实现。<br>图、公式均可离线显示。</p></nav><main><article><header><div class="eyebrow">Transformer · 模型架构</div><h1>Attention 与<br>MHA / MQA / GQA</h1><p>先把一次注意力计算算清楚，再理解多头和共享 KV 头为什么会改变推理成本。</p><p>默认讨论 decoder-only Transformer；不同模型的头数、布局和 kernel 仍需以配置为准。</p></header>{''.join(sections)}<footer>Markdown 为正文源文件，Python 生成插图与阅读版。编写日期：2026-10-15。</footer></article></main></body></html>'''
    (TOPIC/"阅读版.html").write_text(document,encoding="utf-8")


def main():
    ASSETS.mkdir(parents=True,exist_ok=True); configure(); flow(); head_sharing(); cache()
    css=(Path(__file__).resolve().parents[3]/"foundations/tokenization/src/reader.css").read_text(encoding="utf-8")
    reader(css)
    sheet=Image.new("RGB",(1600,800),"#e9edf0")
    for i,path in enumerate(sorted(ASSETS.glob("[0-9][0-9]-*.png"))):
        with Image.open(path) as src:
            image=ImageOps.contain(src.convert("RGB"),(780,380)); sheet.paste(image,((i%2)*800+(800-image.width)//2,(i//2)*400+10))
    sheet.save(ASSETS/"overview.png")
    print("Built 3 figures, formula SVGs, and HTML reader.")

if __name__=="__main__": main()
