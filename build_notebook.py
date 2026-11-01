# -*- coding: utf-8 -*-
"""生成 h3_colab.ipynb：把 h3_app.py 以 base64 内嵌，保证只上传一个文件。"""
import base64
import json
import pathlib

HERE = pathlib.Path(__file__).parent
APP_B64 = base64.b64encode((HERE / "h3_app.py").read_bytes()).decode()

CELLS = []


def md(text):
    CELLS.append(("markdown", text))


def code(text):
    CELLS.append(("code", text))


md("""# MiniMax H3（一条龙）

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lnr095-arch/h3-colab/blob/main/h3_colab.ipynb)

**你不需要会 ComfyUI，也不需要会写 H3 提示词。**

跑完这个笔记本，你会得到一个网页：

1. 你用**人话**写一句"我想要什么效果"
2. AI 按 H3 官方规范把它写成**结构化长提示词**（镜头、时间轴、对白、音效、配乐）
3. 你确认或微调
4. 点生成 → 等 2~6 分钟 → 拿视频

| 项目 | 值 |
|---|---|
| 代码仓库 | [github.com/lnr095-arch/h3-colab](https://github.com/lnr095-arch/h3-colab)（支持热更新） |
| 还需要 | 一个 Gemini API Key（支持 Colab Secrets 或第 7 节填写） |
| 权重体积 | 约 37 GB，首次拉取约 5~15 分钟 |
| 会话上限 | 12 小时，闲置会断 |
| 出片速度 | 5 秒片，A100 约 2~6 分钟 |
| 成片去哪 | 自动存到 `Google Drive / h3-cloud / outputs` |

**顺序**：从上到下，一格格 Shift+Enter（或直接菜单栏「代码执行程序」→「全部运行」）。第 8 节会给你网址。
""")

md("## 1. 环境自检")

code("""!nvidia-smi
import torch

p = torch.cuda.get_device_properties(0)
print(f"torch {torch.__version__} | cuda {torch.version.cuda}")
print(f"显卡: {p.name} | 显存 {p.total_memory / 2**30:.0f} GB | sm_{p.major}{p.minor}")
print("如果显存不是 40/80 GB，说明分到的是 L4，也能跑，只是慢。")
""")

md("""## 2. 挂载 Google Drive

只存成片和日志，**模型不放这里**（Drive 读得太慢，会让启动慢十倍）。
""")

code("""from google.colab import drive
import pathlib

drive.mount('/content/drive')

ROOT = pathlib.Path('/content/drive/MyDrive/h3-cloud')
for sub in ['outputs', 'notes']:
    (ROOT / sub).mkdir(parents=True, exist_ok=True)
print('成片会自动存到:', ROOT / 'outputs')
""")

md("""## 3. 安装 ComfyUI

锁定 v0.30.2（H3 的节点从 0.30.0 起才在内核里）。
""")

code("""%cd /content
!test -d /content/ComfyUI || (git clone --depth 1 --branch v0.30.2 https://github.com/Comfy-Org/ComfyUI || git clone --depth 1 https://github.com/Comfy-Org/ComfyUI ComfyUI)
%cd /content/ComfyUI
!pip install -q -r requirements.txt
!pip install -q "huggingface_hub[cli,hf_transfer]"
!python -c "import torch; print('torch 保持:', torch.__version__)"
""")

md("""## 4. 下载权重（约 37GB）

从 HuggingFace 直拉，别自己上传。

| 文件 | 大小 | 作用 |
|---|---|---|
| `fl2va_pruned_int8_convrot` | 19.5G | 主模型 |
| `qwen3vl_32b_nvfp4_awq` | 14.6G | 文字理解（官方注明不需要 Blackwell） |
| `video_vae_fp16` | 4.9G | 画面解码（官方推荐，彻底解决黑屏 NaN 缺陷） |
| `audio_vae_fp32` | 0.6G | 声音解码 |
| `fl2v_turbo_8step` | — | 加速 LoRA（8 步） |
""")

code("""import os
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

from huggingface_hub import hf_hub_download

MODELS = '/content/ComfyUI/models'
REPO = 'Comfy-Org/MiniMax-H3'
FILES = [
    'diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors',
    'text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors',
    'vae/minimax_h3_video_vae_fp16.safetensors',
    'vae/minimax_h3_audio_vae_fp32.safetensors',
    'loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors',
]
for f in FILES:
    print('↓', f)
    try:
        hf_hub_download(repo_id=REPO, filename=f, local_dir=MODELS)
    except Exception as exc:
        print(f'⚠️ 高速通道下载遇到异常 ({exc})，自动切为标准通道重试...')
        os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
        hf_hub_download(repo_id=REPO, filename=f, local_dir=MODELS)
        os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

!rm -rf ~/.cache/huggingface
!du -sh /content/ComfyUI/models; df -h /content | tail -1
""")

md("""## 5. 装视频读取插件

界面只用来做文生视频和首尾帧，其实这个插件是给参考视频（R2V）预留的。装上不碍事。
""")

code("""%cd /content/ComfyUI/custom_nodes
!test -d ComfyUI-VideoHelperSuite || git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite
!pip install -q -r ComfyUI-VideoHelperSuite/requirements.txt
%cd /content/ComfyUI
!python -c "import gradio, requests; print('gradio', gradio.__version__, '| requests ok')"
""")

md("""## 6. 启动后台引擎（ComfyUI）

跑完就忘掉它。它只在 `127.0.0.1:8188` 上工作，不对外。

日志在 `/content/comfy.log`。
""")

code("""import subprocess, time, os, torch, requests

os.chdir('/content/ComfyUI')
vram = torch.cuda.get_device_properties(0).total_memory / 2**30
flag = '--highvram' if vram >= 70 else '--normalvram'
print(f'显存 {vram:.0f} GB → 启动参数 {flag}')

!pkill -f "main.py --listen" || true
time.sleep(1)

# 使用 nohup 独立守护进程启动，防止 Python 垃圾回收导致管道中断
!nohup python main.py --listen 127.0.0.1 --port 8188 {flag} > /content/comfy.log 2>&1 &

print('等待 ComfyUI 引擎启动就绪...')
ready = False
for _ in range(35):
    time.sleep(2)
    try:
        if requests.get('http://127.0.0.1:8188/system_stats', timeout=2).status_code == 200:
            ready = True
            break
    except Exception:
        pass

if ready:
    print('✅ ComfyUI 引擎已就绪 (127.0.0.1:8188)')
else:
    print('⚠️ ComfyUI 尚未完成就绪检查，查看日志最新输出：')
    print(open('/content/comfy.log').read()[-800:])
""")

md("""## 7. 启动生成界面

这一步会通过 Git 动态拉取最新业务代码并运行。
如果我们在本地修复了 Bug，直接重跑本单元格即可**秒级热更新**，无需重新下载 37GB 权重！

**配置 Gemini API Key**（二选一）：
* **推荐**：在 Colab 左侧「🔑 机密/Secrets」添加名称为 `GEMINI_API_KEY` 的密钥，一劳永逸。
* **或者**：直接粘到下方代码里的 `GEMINI_API_KEY = ""` 引号中。
""")

code("""import base64, pathlib, subprocess, time, os
import requests

# 1. 尝试从 Colab Secrets 自动读取（或下方手动填写）
GEMINI_API_KEY = ""
try:
    from google.colab import userdata
    GEMINI_API_KEY = userdata.get('GEMINI_API_KEY') or GEMINI_API_KEY
except Exception:
    pass

# 2. 从 GitHub 热拉取最新代码（修 Bug 秒生效，无需重新打包笔记本）
repo_dir = pathlib.Path('/content/h3-colab')
app_script = '/content/h3-colab/h3_app.py'
try:
    if not (repo_dir / '.git').exists():
        !rm -rf /content/h3-colab
        !git clone https://github.com/lnr095-arch/h3-colab.git /content/h3-colab
    else:
        !cd /content/h3-colab && git pull
except Exception as e:
    print(f'⚠️ Git 拉取遇到异常 ({e})，使用内嵌备用代码启动...')
    APP_B64 = "__APP_B64__"
    pathlib.Path('/content/h3_app.py').write_bytes(base64.b64decode(APP_B64))
    app_script = '/content/h3_app.py'

# 3. 杀掉可能残留的旧进程，避免端口被占
!pkill -f h3_app.py || true
time.sleep(2)

env = os.environ.copy()
if GEMINI_API_KEY.strip():
    env["GEMINI_API_KEY"] = GEMINI_API_KEY.strip()

app_log = open('/content/app.log', 'w')
subprocess.Popen(['python', app_script], env=env,
                 stdout=app_log, stderr=subprocess.STDOUT)

ok = False
for _ in range(30):
    time.sleep(2)
    try:
        if requests.get('http://127.0.0.1:7860', timeout=3).status_code == 200:
            ok = True
            break
    except Exception:
        pass

if ok:
    print('✅ 界面已就绪（127.0.0.1:7860）')
    gradio_url = None
    for _ in range(10):
        try:
            m = re.search(r'https://[a-zA-Z0-9-]+\\.gradio\\.live', open('/content/app.log').read())
            if m:
                gradio_url = m.group(0)
                break
        except Exception:
            pass
        time.sleep(1)
    if gradio_url:
        print('=' * 62)
        print('🎉 官方直连公网地址：', gradio_url)
        print('=' * 62)
    print('提示：也可以继续跑第 8 节获取 Cloudflare 备用线路')
else:
    print('❌ 界面启动失败，日志最后部分：')
    print('-' * 60)
    try:
        print(open('/content/app.log').read()[-3000:])
    except Exception:
        pass
    print('-' * 60)
""")

md("""## 8. 拿到网址

跑完会打印一个 `https://xxxx.trycloudflare.com`。**这就是你的生成页面**，手机电脑都能开。

地址是临时的、随机的，会话结束就失效。地址会自动同步备份一份到 Google Drive。
""")

code("""import subprocess, time, re, requests, pathlib

# 先确认界面活着，否则隧道开了也是 502
try:
    code_ = requests.get('http://127.0.0.1:7860', timeout=5).status_code
    print('本机界面应答:', code_)
except Exception as exc:
    print('⚠️ 本机界面没有应答，先把第 7 节跑通，否则打开一定是 502:', exc)

!wget -q -O /content/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
!chmod +x /content/cloudflared
!pkill -f cloudflared || true
!rm -f /content/tunnel.log
time.sleep(2)

subprocess.Popen(
    ['/content/cloudflared', 'tunnel', '--url', 'http://127.0.0.1:7860',
     '--no-autoupdate', '--logfile', '/content/tunnel.log'],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)

url = None
for _ in range(45):
    time.sleep(2)
    try:
        matches = re.findall(r'https://[-a-z0-9]+\\.trycloudflare\\.com', open('/content/tunnel.log').read())
        if matches:
            url = matches[-1]
            break
    except FileNotFoundError:
        pass

if url:
    try:
        drive_note = pathlib.Path('/content/drive/MyDrive/h3-cloud/notes')
        drive_note.mkdir(parents=True, exist_ok=True)
        (drive_note / 'last_url.txt').write_text(url)
    except Exception:
        pass

print('\\n' + '=' * 62)
print('  打开这个地址:', url or '没拿到，看 /content/tunnel.log')
print('=' * 62)
print('如果打开是 502：说明隧道通了但界面没应答 → 回第 7 节重跑')
""")

md("""## 9. 收工

用完一定要跑这一格。不跑的话 Colab 会一直占着显卡计费（虽然断开也会停）。
""")

code("""!pkill -f cloudflared || true
!pkill -f h3_app.py || true
!pkill -f "main.py --listen" || true
!cp /content/comfy.log "/content/drive/MyDrive/h3-cloud/notes/comfy-last.log" 2>/dev/null || true
!cp /content/app.log "/content/drive/MyDrive/h3-cloud/notes/app-last.log" 2>/dev/null || true
print('已停止，日志备份到 Drive/notes/')
""")

md("""## 10. 诊断（出问题时跑这一格，把输出发我）

什么都不用改，直接跑，把打印出来的内容截图或复制给我。
""")

code("""print('--- 界面日志（最后 2500 字） ---')
try:
    print(open('/content/app.log').read()[-2500:])
except Exception as exc:
    print('没有 app.log:', exc)

print('\\n--- 引擎日志（最后 1500 字） ---')
try:
    print(open('/content/comfy.log').read()[-1500:])
except Exception as exc:
    print('没有 comfy.log:', exc)

print('\\n--- 端口监听 ---')
!ss -ltnp 2>/dev/null | grep -E '7860|8188' || echo '7860/8188 都没有监听'

print('\\n--- 进程 ---')
!ps -eo pid,etime,cmd 2>/dev/null | grep -E 'h3_app|main.py|cloudflared' | grep -v grep || echo '没有相关进程'

print('\\n--- 显卡 ---')
!nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
""")

md("""## 卡住了看这里

**第 3 节报红 / pip 冲突**
再跑一次。Colab 的预装包偶尔会打架，第二次通常就好。

**第 4 节某个文件卡住不动**
点这格的「停止」，重跑该格。已下完的不会重下。

**第 6 节出现 `CUDA out of memory`**
分到的是 L4。把界面里的分辨率换成 768×768、时长 4 秒、步数 12 再试。

**第 8 节等 90 秒还没网址**
重跑第 8 节。还不行就看 `/content/tunnel.log`。

**网址打开是 502**
界面还在启动。等 30 秒刷新。

**生成时报错 `ComfyUI 拒收任务`**
看第 6 节日志末尾，通常是模型没下完（第 4 节没跑完）。

**点「帮我写提示词」报 `需要 Gemini API Key`**
第 7 节没填 key。回去填上，重跑第 7、8 节。

**报 `Gemini 400` / `API key not valid`**
key 复制错了（多了空格或换行），重新复制一次。

**报 `Gemini 429`**
免费额度用完了，等明天。或在界面「模型」里指定一个更轻的（如 gemini-flash 系列）。

**提示词是英文，看不懂**
正常。H3 只吃英文提示词，别改成中文再喂进去。

**改了时长，出的片和提示词对不上**
时长变了要重新点一次「帮我写提示词」，镜头时间轴是按参数写进提示词的。

**生成的视频很糊 / 结构怪**
- 步数提到 25
- 关掉「加速模式」
- 换个更具体的提示词，写清主体、动作、镜头、声音

**想复现同一条**
把种子从 -1 改成一个固定数字。

---

## 关于授权（别忽略）

ComfyUI 官方文档写明：**本地生成内容的商用需要 MiniMax 商业许可**（Comfy 是唯一官方转售方），Comfy Cloud 上生成的才自带商用权。

发 B 站纯分享一般不涉及；**开了创作激励、接广告、带货就可能算商用**。这条跟用不用云无关。
""")

nb = {
    "cells": [],
    "metadata": {
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

for kind, src in CELLS:
    if kind == "code":
        src = src.replace("__APP_B64__", APP_B64)
    cell = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    nb["cells"].append(cell)

out = HERE / "h3_colab.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"wrote {out} ({len(CELLS)} cells, app embedded {len(APP_B64)} b64 chars)")
