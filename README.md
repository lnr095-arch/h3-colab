# H3 上云方案

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lnr095-arch/h3-colab/blob/main/h3_colab.ipynb)

> 用 AI Pro 的 200 Colab 算力单位跑 MiniMax H3。
> **你只需要用人话描述想要什么**——提示词由 Gemini 按 H3 官方规范代写。

---

## 架构

```
你：写一句人话「雨夜赛博朋克街道，镜头从地面升起」
      │
      ▼
Gemini API  ← 系统提示词 = MiniMax 官方 H3 提示词规范（压缩版）
      │        输出：integrated_multimodal_description / overall_soundscape / non_diegetic_music
      ▼
你：看一眼，改或重写
      │
      ▼
ComfyUI :8188（无界面，只监听 127.0.0.1）→ 出片
      │
      ▼
Google Drive / h3-cloud / outputs（自动备份）
```

界面只有一层，跑在 Colab 的 `:7860` 上，通过临时隧道给你访问。

---

## 为什么提示词必须由 AI 写

H3 不是"随便写句话"的模型。它的提示词是**结构化规范**：

```
integrated_multimodal_description: [Shot 1] Live-action, cinematic, a medium-wide shot frames...
[Shot 2] At 00:03.500, the camera cuts to...

overall_soundscape: Steady rain taps against the café windows...

non_diegetic_music: Sparse piano notes at a slow tempo...
```

硬规则（来自官方 guide）：

- 必须英文
- 镜头要有编号，第二镜起必须带**严格递增**的切换时间
- 镜头运动要写成 `motion type + amplitude + speed` 的自然语言，比如 `The camera pushes in with small amplitude at slow speed`
- 说话人要有稳定 ID `(S1)`，台词放 `<d>[English] ...</d>`，且原话不能翻译
- 屏幕上的文字要用英文双引号、原样保留
- **不能写否定**（没有 negative 分支）
- `overall_soundscape` 1~4 句，`non_diegetic_music` 1~3 句，都不能和画面对白重复

这种东西人肉写一天能写几条？所以 `h3_app.py` 里内嵌了一份压缩版规范作为 Gemini 的 system prompt，你只管说想要什么。

官方其实也提供了同样思路的 [h3-prompt-writing skill](https://github.com/MiniMax-AI/MiniMax-H3/tree/main/skills)，我把它要遵守的规则直接写进了系统提示词。

---

## 文件

| 文件 | 作用 |
|---|---|
| [h3_colab.ipynb](h3_colab.ipynb) | **唯一需要上传的文件**，9 节，界面程序已内嵌 |
| [walkthrough.md](walkthrough.md) | 手把手说明（第一次看这个） |
| [h3_app.py](h3_app.py) | 界面 + 提示词改写 + 节点图，全在这一个文件 |
| [build_notebook.py](build_notebook.py) | 构建脚本，改完重跑即可重新生成 ipynb |

---

## 几个关键决策

### 1. 提示词由 LLM 写，不由人写

这是整个方案的核心。界面分成「① 你的想法 → ② 帮我写提示词 → ③ 可编辑的成品提示词 → ④ 生成」，而不是甩一个空框让人自己填。

Gemini 的模型 ID 是**运行时自动发现**的（调 `models.list`，按版本号和 pro/flash 打分选最强的），不写死——避免哪天模型改名就崩。

### 2. 生成过程用生成器推流

2~6 分钟没有任何反馈，用户会以为卡死。`generate()` 写成生成器，每 3 秒 `yield` 一次状态（排队位置 / 已用秒数）。默认的 Gradio 按钮会一直转圈，现在能看到进度。

### 3. 成片自动进 Drive

下载完直接 `shutil.copy2` 到 `Drive/h3-cloud/outputs/`，失败只静默跳过。这是最容易丢片的地方。

### 4. 模型不放 Drive

| 方案 | 每次会话启动 |
|---|---|
| 从 Drive 读 37GB | 10~25 分钟（随机读慢 + IO 限流） |
| 从 HF 直拉 37GB | **5~15 分钟** |

### 5. 界面程序 base64 内嵌

只上传一个 `.ipynb`。构建时把 `h3_app.py` 编码进第 7 节，运行时解码写盘再执行。已校验内嵌内容与源文件 sha256 一致。

---

## 额度怎么花

| 阶段 | 用哪张卡 | 干什么 |
|---|---|---|
| 调试 | **L4** 24GB | 试想法、试提示词，便宜 |
| 出片 | **A100** 80GB | 参数定了批量出，快约 3 倍 |

切换「代码执行程序 → 更改运行时类型」，不用重跑第 1~5 节。第 6 节按显存自动选 `--highvram`（≥70GB）。

Gemini 那边消耗的是 AI Studio 的免费额度，跟 Colab 算力单位无关，用量极小。

---

## 参数基准

| 参数 | 值 |
|---|---|
| 原生分辨率 | 1344×768（16:9，768 短边 = 1MP） |
| 帧数 | 必须 17k+5 @24fps；124 帧 ≈ 5.17 秒（官方默认） |
| 步数 | 默认 20；turbo 8；画质 25 |

不要拉到 1080p/2K 直接采样——超出训练分布，只会重复纹理 + 闪烁，注意力成本还平方级上涨。

---

## 已知限制

| 限制 | 影响 |
|---|---|
| 会话最长 12 小时 | 长批次被腰斩 |
| 闲置超时断连 | 挂着走开就没了 |
| 磁盘临时 | 权重每次重下 |
| 无认证 | 隧道地址泄露 = 谁都能用你的额度 |
| 资源不保证 | 高峰期分不到 A100 |
| Gemini 免费额度 | 429 时等明天 |

**这是沙盒，不是服务器。**

---

## 合规

> Commercial use of locally generated outputs requires a MiniMax commercial license, available through Comfy, the only official reseller. Generations on Comfy Cloud already include commercial rights.

本地或 Colab 生成的片子**商用需 MiniMax 商业许可**。发 B 站纯分享一般不涉及；开了创作激励、接广告、带货可能算商用。

---

## 附：本地也能用

`h3_app.py` 只通过 HTTP 对接 ComfyUI，不关心跑在哪：

```powershell
$env:COMFYUI_URL="http://127.0.0.1:8188"
$env:COMFY_DIR="F:\ComfyUI-aki-v3.2\ComfyUI"
$env:H3_DRIVE_OUT="D:\h3-output"
$env:GEMINI_API_KEY="AIza..."
F:\ComfyUI-aki-v3.2\python\python.exe h3_app.py
```

比你本机 codex 写的 `h3_webui.py` 干净：没有硬编码路径、提示词由 AI 代写、成片自动归档。
