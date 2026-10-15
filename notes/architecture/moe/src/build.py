"""Build simple MoE diagrams and an offline reader."""
from pathlib import Path
import hashlib, html, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, mathtext
from matplotlib.patches import FancyBboxPatch, Rectangle
import markdown
from PIL import Image, ImageOps

TOPIC=Path(__file__).resolve().parents[1]; ASSETS=TOPIC/"assets"
BLUE,ORANGE,GREEN,PURPLE="#24649a","#ce7739","#34857b","#7962ac"
CHAPTERS=[
 ("01-为什么需要MoE.md","chapter-01","01 · 为什么需要 MoE"),
 ("02-Router与Top-k路由.md","chapter-02","02 · Router 与 Top-k"),
 ("03-负载均衡损失.md","chapter-03","03 · 负载均衡损失"),
 ("04-细粒度专家与共享专家.md","chapter-04","04 · 细粒度与共享专家"),
 ("05-训练与推理实现.md","chapter-05","05 · 训练与推理实现"),
]

def configure():
    font=Path("C:/Windows/Fonts/msyh.ttc")
    if font.exists(): font_manager.fontManager.addfont(str(font)); family=font_manager.FontProperties(fname=str(font)).get_name()
    else: family="DejaVu Sans"
    plt.rcParams.update({"font.family":family,"font.size":12,"axes.unicode_minus":False,"svg.fonttype":"path","axes.spines.top":False,"axes.spines.right":False,"axes.edgecolor":"#c1c9d1","text.color":"#263440","xtick.color":"#56616d","ytick.color":"#56616d","figure.facecolor":"white","savefig.dpi":180})

def save(fig,name,title,note):
    fig.suptitle(title,x=.055,y=.97,ha="left",fontsize=18,fontweight="bold"); fig.text(.055,.035,note,color="#697582",fontsize=10); fig.subplots_adjust(left=.09,right=.96,bottom=.19,top=.79)
    for ext in ("png","svg"): fig.savefig(ASSETS/f"{name}.{ext}")
    plt.close(fig)

def dense_moe():
    fig,axes=plt.subplots(1,2,figsize=(11,5.1))
    for ax,title,experts,active,color in [(axes[0],"Dense FFN",1,1,BLUE),(axes[1],"Sparse MoE",4,2,GREEN)]:
        ax.set(xlim=(-.3,5.2),ylim=(-.2,4.2)); ax.axis("off"); ax.set_title(title,pad=12,fontsize=15,fontweight="bold")
        ax.add_patch(FancyBboxPatch((.2,1.65),.9,.7,boxstyle="round,pad=.04",fc="#f5f8fa",ec=BLUE)); ax.text(.65,2.0,"token x",ha="center",va="center")
        ax.annotate("",(1.8,2.0),(1.15,2.0),arrowprops={"arrowstyle":"->","color":"#8999a7"})
        ys=[2.0] if experts==1 else [3.1,2.0,.9,.2]
        for i,y in enumerate(ys):
            on=experts==1 or i<active; ec=color if on else "#c8d1d8"; fc="#edf6f2" if on else "#f5f6f7"
            ax.add_patch(FancyBboxPatch((1.9,y-.3),1.35,.6,boxstyle="round,pad=.04",fc=fc,ec=ec,lw=1.5)); ax.text(2.575,y,f"E{i+1}",ha="center",va="center",color="#40515e" if on else "#9ba6af")
            ax.annotate("",(3.65,2.0),(3.3,y),arrowprops={"arrowstyle":"->","color":ec,"lw":1.4,"alpha":.8})
        ax.add_patch(FancyBboxPatch((3.7,1.65),1.1,.7,boxstyle="round,pad=.04",fc="#f5f8fa",ec=PURPLE)); ax.text(4.25,2.0,"加权合并",ha="center",va="center",fontsize=10)
        if experts>1: ax.text(2.55,3.72,"只激活 2 个专家",ha="center",fontsize=10,color=GREEN)
    save(fig,"01-dense-vs-moe","稠密 FFN 与稀疏 MoE 的计算路径","示意：MoE 总共有 4 个专家，但这个 token 只执行其中 2 个。")

def router_capacity():
    fig,axes=plt.subplots(1,2,figsize=(11,5.2))
    ax=axes[0]; ax.set(xlim=(-.5,7.5),ylim=(-.5,4.7)); ax.axis("off"); ax.set_title("Top-k Router",fontsize=15,fontweight="bold")
    for i in range(6):
        ax.add_patch(Rectangle((i,.1),.7,.55,fc="#eef4f7",ec=BLUE)); ax.text(i+.35,.375,f"t{i}",ha="center",va="center",fontsize=10)
        target=(i*2)%4; ax.annotate("",(target+1.0,2.75),(i+.35,.68),arrowprops={"arrowstyle":"->","color":GREEN,"lw":1.2})
    for i in range(4):
        ax.add_patch(FancyBboxPatch((i+.5,2.75),1,.65,boxstyle="round,pad=.04",fc="#edf6f2",ec=GREEN)); ax.text(i+1,3.08,f"E{i}",ha="center",va="center")
    ax.text(3.0,4.1,"每个 token 选择 k 个专家",ha="center",color=PURPLE)
    ax=axes[1]; vals=[2,2,5,1]; bars=ax.bar(range(4),vals,color=[BLUE,BLUE,ORANGE,BLUE]); ax.axhline(3.2,color=PURPLE,ls="--",label="容量 C")
    ax.set(xticks=range(4),xticklabels=["E0","E1","E2","E3"],ylabel="收到的 token 副本数",ylim=(0,6)); ax.legend(frameon=False); ax.grid(axis="y",alpha=.17)
    ax.annotate("溢出",(2,5),xytext=(2.35,5.2),arrowprops={"arrowstyle":"->","color":ORANGE},color=ORANGE)
    save(fig,"02-router-capacity","Router 选择之后，还要处理容量和溢出","左：token 按目标专家分桶；右：超过容量 C 的热门专家会产生溢出。")

def shared_fine():
    fig,ax=plt.subplots(figsize=(11,5.2)); ax.set(xlim=(0,11),ylim=(0,5)); ax.axis("off")
    def box(x,y,w,h,label,ec,fc="#f5f8fa"):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.04",fc=fc,ec=ec,lw=1.4)); ax.text(x+w/2,y+h/2,label,ha="center",va="center",fontsize=11)
    box(.3,2.05,1.2,.7,"token x",BLUE); ax.annotate("",(2.15,2.4),(1.55,2.4),arrowprops={"arrowstyle":"->","color":"#8999a7"})
    box(2.2,3.1,2.0,.7,"shared expert",ORANGE,"#fff5ed"); box(2.2,.95,2.0,.7,"Router",PURPLE,"#f4f0fa")
    for y in (3.45,1.3): ax.annotate("",(4.75,y),(4.25,y),arrowprops={"arrowstyle":"->","color":ORANGE if y>2 else PURPLE})
    for i in range(3): box(4.95,.65+i*1.15,1.45,.7,f"细粒度 E{i+1}",GREEN,"#edf6f2")
    ax.annotate("",(4.95,1.3),(4.25,1.3),arrowprops={"arrowstyle":"->","color":PURPLE})
    box(7.25,2.05,1.55,.7,"相加 / 合并",BLUE); ax.annotate("",(7.2,2.4),(6.4,2.4),arrowprops={"arrowstyle":"->","color":"#8999a7"}); ax.annotate("",(7.2,2.4),(4.2,3.45),arrowprops={"arrowstyle":"->","color":ORANGE,"connectionstyle":"arc3,rad=-.25"})
    box(9.15,2.05,1.2,.7,"输出 y",BLUE); ax.annotate("",(9.1,2.4),(8.85,2.4),arrowprops={"arrowstyle":"->","color":"#8999a7"})
    save(fig,"03-shared-fine-grained","共享专家与细粒度路由专家可以组合使用","共享分支对所有 token 执行；路由专家由 Router 选择；图为结构示意。")

def formula(match):
    expression=" ".join(match.group(1).split()); digest=hashlib.sha256(expression.encode()).hexdigest()[:12]; folder=ASSETS/"equations"; folder.mkdir(exist_ok=True)
    rendered=expression.replace(r"\operatorname", "").replace(r"\left", "").replace(r"\right", "").replace(r"\lceil", "[").replace(r"\rceil", "]")
    with plt.rc_context({"savefig.transparent":True}): mathtext.math_to_image(f"${rendered}$",folder/f"{digest}.svg",prop=font_manager.FontProperties(size=18),format="svg",color="#25313d")
    return f'\n\n<div class="equation"><img src="assets/equations/{digest}.svg" alt="{html.escape(expression,quote=True)}"></div>\n\n'

def reader(css):
    sections=[]
    for filename,anchor,_ in CHAPTERS:
        text=(TOPIC/filename).read_text(encoding="utf-8"); text=re.sub(r"\$\$\s*\n(.*?)\n\s*\$\$",formula,text,flags=re.S)
        for source,target,_ in CHAPTERS: text=text.replace(f"]({source})",f"](#{target})")
        body=markdown.markdown(text,extensions=["tables","fenced_code","toc"]); body=re.sub(r'id="([^"]+)"',lambda m:f'id="{anchor}-{m.group(1)}"',body)
        def fig(m):
            src=re.search(r'src="([^"]+)"',m.group(1)).group(1); return f'<figure class="illustration"><a href="{src[:-4]}.svg" target="_blank">{m.group(1)}</a><figcaption>点击插图查看可缩放的 SVG 大图</figcaption></figure>'
        body=re.sub(r'<p>(<img [^>]+>)</p>',fig,body); sections.append(f'<section class="chapter" id="{anchor}">{body}</section>')
    links="".join(f'<a href="#{a}">{label}</a>' for _,a,label in CHAPTERS)
    doc=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MoE 架构 · 阅读版</title><style>{css}</style></head><body><nav aria-label="章节目录"><div class="brand">大模型 · 模型架构</div>{links}<hr><a href="README.md">专题源文件</a><a href="../../../README.md">资料总入口</a><p>先看 Router 选择，再看容量和均衡。<br>图、公式均可离线显示。</p></nav><main><article><header><div class="eyebrow">Transformer · 模型架构</div><h1>MoE 架构<br>Router、负载均衡与专家组织</h1><p>理解 token 为什么只经过少数专家，以及这件事如何影响训练稳定性、通信和推理成本。</p><p>默认讨论稀疏 MoE FFN；具体模型的路由、容量和专家配置仍需核对实现。</p></header>{''.join(sections)}<footer>Markdown 为正文源文件，Python 生成插图与阅读版。编写日期：2026-10-15。</footer></article></main></body></html>'''
    (TOPIC/"阅读版.html").write_text(doc,encoding="utf-8")

def main():
    ASSETS.mkdir(parents=True,exist_ok=True); configure(); dense_moe(); router_capacity(); shared_fine(); css=(Path(__file__).resolve().parents[3]/"foundations/tokenization/src/reader.css").read_text(encoding="utf-8"); reader(css)
    sheet=Image.new("RGB",(1600,800),"#e9edf0")
    for i,path in enumerate(sorted(ASSETS.glob("[0-9][0-9]-*.png"))):
        with Image.open(path) as src:
            image=ImageOps.contain(src.convert("RGB"),(780,380)); sheet.paste(image,((i%2)*800+(800-image.width)//2,(i//2)*400+10))
    sheet.save(ASSETS/"overview.png"); print("Built 3 figures, formula SVGs, and HTML reader.")

if __name__=="__main__": main()
