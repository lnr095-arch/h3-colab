# -*- coding: utf-8 -*-
"""
MiniMax H3 生成界面 —— 你描述想要什么，AI 写提示词，然后出片。

流程：
    1. 你写一句人话（"想要一个雨夜赛博朋克街头，镜头从地面往上摇"）
    2. 点「帮我写提示词」→ Gemini 按 H3 官方规范改写成结构化长提示词
    3. 你可以直接改，或点「重写」
    4. 点「生成视频」→ ComfyUI 执行

后台对接 127.0.0.1:8188 的 ComfyUI，节点图内置在代码里。
"""
import os
import re
import time
import uuid
import json
import shutil
import pathlib

import requests
import gradio as gr

COMFY = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")
COMFY_DIR = pathlib.Path(os.environ.get("COMFY_DIR", "/content/ComfyUI"))
OUT_DIR = COMFY_DIR / "output"
IN_DIR = COMFY_DIR / "input"
DRIVE_OUT = pathlib.Path(os.environ.get("H3_DRIVE_OUT", "/content/drive/MyDrive/h3-cloud/outputs"))

GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"

DIT = os.environ.get("H3_DIT", "minimax_h3_fl2va_pruned_int8_convrot.safetensors")
ENC = os.environ.get("H3_ENC", "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")
VVAE = os.environ.get("H3_VVAE", "minimax_h3_video_vae_int8_convrot.safetensors")
AVAE = os.environ.get("H3_AVAE", "minimax_h3_audio_vae_fp32.safetensors")
TURBO_LORA = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"

RATIOS = {
    "16:9 横屏": (1344, 768),
    "9:16 竖屏": (768, 1344),
    "1:1 方形": (768, 768),
    "4:3 横屏": (1024, 768),
}

# ---------------------------------------------------------------------------
# H3 提示词规范（压缩自 MiniMax 官方 VIDEO_PROMPT_WRITING_GUIDE_base_en.md）
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = r"""You are MiniMax H3 prompt writer. Rewrite the user's idea into a single English
video-generation prompt that strictly follows H3's format.

OUTPUT FORMAT — output only the prompt, no explanation, no markdown fences.

[Optional mode line — only when a keyframe image is given]
<blank line>
integrated_multimodal_description: [Shot 1] ...
<blank line>
overall_soundscape: ...
<blank line>
non_diegetic_music: ...

MODE LINES (verbatim, first line of the prompt, then ONE blank line):
- first frame only (I2VA):
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.
- first and last frame (FL2VA):
How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot N) aligns with the S.SS-second mark of the target video.
- last frame only (L2VA):
How the reference pictures align with the target video — <Picture 1> (from [Shot N]) aligns with the S.SS-second mark of the target video.
Replace N with the real final shot number and S.SS with the real duration to exactly two decimals.
- no image (T2VA): no mode line, start directly at integrated_multimodal_description.

integrated_multimodal_description
- Everything visible and audible on the timeline, one flowing block.
- Start [Shot 1] with style + initial composition, e.g. "Live-action, cinematic, a medium-wide shot frames..."
- Do NOT put a timestamp on Shot 1. Later shots: "[Shot 2] At 00:03.500, the camera cuts to ..."
  Cut times strictly increase and stay inside the duration. Use a single time, never a range.
- Cut verbs: "the camera cuts to" / "the shot cuts to" / "the shot transitions to" /
  "the shot changes to" / "the shot switches to". Change shot only when subject, space,
  state, viewpoint or time changes; otherwise use camera motion.
- Camera motion = motion type + optional amplitude + optional speed, written as natural prose:
  type: Zoom In/Out, Push In/Pull Out, Pan Left/Right, Truck Left/Right, Tilt Up/Down,
  Pedestal Up/Down, Arc Shot, Tracking Shot, Static Shot, Shake Slightly/Strongly, POV,
  Roll Clockwise/Counterclockwise.
  amplitude: "with small amplitude" / "with large amplitude".
  speed: "at slow speed" / "at fast speed".
  Example: "The camera pushes in with small amplitude at slow speed toward the letter."
- Speakers: stable IDs (S1), (S2); together "(S1,S2)". Give identity cues when first heard
  (age, gender, on/off-screen, pitch, timbre, rate, accent). ID, action and delivery stay
  OUTSIDE <d>; inside <d> put only the language tag and the exact spoken words:
  "The young woman with a quiet, breathy voice (S1) says: <d>[English] I get off at the next station.</d>"
  Keep original punctuation, no translation, no ellipsis spam, end with . ? or !.
- Voiceover: exact phrase "says in an off-screen voiceover", then state the on-screen
  character's lips remain completely closed.
- Dialogue crossing a cut: put <scenetrans> at both connecting points and say the audio
  continues across the cut. Truncated speech: <cutoff>.
- On-screen text: English double quotes, verbatim, untranslated:
  A red neon sign reading "营业中" glows above the doorway.
- Diegetic music/instruments the characters can hear belong HERE, not in non_diegetic_music.

overall_soundscape
- 1-4 English sentences, one paragraph: ambience, physical action sounds, non-verbal human
  sounds (wind, rain, traffic, footsteps, fabric, impacts, breathing, laughter).
- Do NOT repeat dialogue, singing or diegetic music here.
- Use "N/A" only if the user explicitly asks for total silence.

non_diegetic_music
- 1-3 English sentences: score only, music the characters cannot hear.
- Describe instrumentation, tempo, rhythm, dynamics. No abstract mood words.
- Use "N/A" when there is no non-diegetic music.

HARD RULES
- English only.
- No negative prompts: there is no negative branch. Never write "no ...", "without ...",
  "avoid ...". Describe what SHOULD appear.
- The final shot must end at the requested duration.
- Respect the user's language for any spoken line but keep the <d> language tag correct.
"""


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------
def gemini_models(key):
    r = requests.get(f"{GEMINI_BASE}/models", params={"key": key, "pageSize": 200}, timeout=30)
    r.raise_for_status()
    names = []
    for m in r.json().get("models", []):
        if "generateContent" not in (m.get("supportedGenerationMethods") or []):
            continue
        name = m.get("name", "").replace("models/", "")
        if any(bad in name for bad in ("embedding", "aqa", "tts", "image", "vision", "veo")):
            continue
        names.append(name)
    return names


def score_model(name):
    s = 0
    if "3.8" in name or "3-8" in name:
        s += 500
    elif "3" in name:
        s += 300
    m = re.search(r"gemini-(\d+)", name)
    if m:
        s += int(m.group(1)) * 100
    if "flash" in name:
        s += 60
    elif "pro" in name:
        s += 30
    if "exp" in name or "preview" in name:
        s -= 5
    if "latest" in name:
        s += 3
    return s


def pick_model(key, preferred=""):
    if preferred and preferred.strip():
        return preferred.strip()
    try:
        names = gemini_models(key)
        for n in names:
            if "3.8" in n and "flash" in n:
                return n
        for n in names:
            if "3" in n and "flash" in n:
                return n
        if names:
            return sorted(names, key=score_model)[-1]
    except Exception:
        pass
    return "gemini-3.8-flash"


def call_gemini(key, model, user_msg, temperature=0.7):
    # 配置高思考 (thinkingLevel: HIGH) 与充足 token
    body = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": user_msg}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": 8192,
            "thinkingConfig": {
                "thinkingLevel": "HIGH",
            },
        },
    }
    url = f"{GEMINI_BASE}/models/{model}:generateContent"
    r = requests.post(url, params={"key": key}, json=body, timeout=180)

    # 兼容性降级：若特定网关要求数值型 thinkingBudget
    if r.status_code == 400 and "thinking" in r.text.lower():
        body["generationConfig"]["thinkingConfig"] = {"thinkingBudget": 24576}
        r = requests.post(url, params={"key": key}, json=body, timeout=180)

    # 兼容性回退：若当前账号尚未激活 3.8-flash，自动平滑切至 2.5-flash
    if r.status_code in (404, 400) and "3.8" in model:
        print("gemini-3.8-flash 未激活，自动切换至备选高速推理模型...")
        return call_gemini(key, "gemini-2.5-flash", user_msg, temperature)

    if r.status_code != 200:
        raise RuntimeError(f"Gemini {r.status_code}: {r.text[:300]}")
    cands = r.json().get("candidates") or []
    if not cands:
        raise RuntimeError("Gemini 没有返回内容（可能被安全过滤了）")
    parts = cands[0].get("content", {}).get("parts") or []

    # 过滤掉内部思考 (thought: true)，只把最终纯净的 H3 提示词送入渲染
    output_parts = [p.get("text", "") for p in parts if not p.get("thought", False)]
    if not output_parts:
        output_parts = [p.get("text", "") for p in parts]
    text = "".join(output_parts).strip()
    return re.sub(r"^```[a-z]*\n|\n```$", "", text).strip()


# ---------------------------------------------------------------------------
# 帧数 / 节点图
# ---------------------------------------------------------------------------
def snap_frames(seconds, fps=24):
    """H3 的 length 必须落在 17k+5 上。124 帧 ≈ 5.17 秒（官方默认）。"""
    frames = seconds * fps
    k = max(round((frames - 5) / 17), 0)
    return 17 * k + 5


def mode_of(first, last):
    if first and last:
        return "FL2VA"
    if first:
        return "I2VA"
    if last:
        return "L2VA"
    return "T2VA"


def stage_image(path):
    src = pathlib.Path(path)
    name = f"h3_in_{uuid.uuid4().hex[:8]}{src.suffix.lower() or '.png'}"
    IN_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, IN_DIR / name)
    return name


def build_graph(prompt, width, height, length, steps, seed, turbo, first, last):
    g = {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": DIT, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": ENC, "type": "minimax"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VVAE}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": AVAE}},
    }
    model_src = "1"
    if turbo:
        g["100"] = {"class_type": "LoraLoaderModelOnly",
                    "inputs": {"model": ["1", 0],
                               "lora_name": TURBO_LORA,
                               "strength_model": 1.0}}
        model_src = "100"

    h3 = {"clip": ["2", 0], "vae": ["3", 0], "prompt": prompt,
          "width": int(width), "height": int(height), "length": int(length)}
    if first:
        g["200"] = {"class_type": "LoadImage", "inputs": {"image": stage_image(first)}}
        h3["first_frame"] = ["200", 0]
    if last:
        g["201"] = {"class_type": "LoadImage", "inputs": {"image": stage_image(last)}}
        h3["last_frame"] = ["201", 0]
    g["5"] = {"class_type": "MiniMaxH3ImageToVideo", "inputs": h3}

    # 官方标准 MiniMax H3 采样管线（解决 euler 导致的纯黑屏问题）
    g["10"] = {"class_type": "BasicGuider", "inputs": {
        "model": [model_src, 0],
        "conditioning": ["5", 0],
    }}
    g["11"] = {"class_type": "KSamplerSelect", "inputs": {
        "sampler_name": "res_multistep",
    }}
    g["12"] = {"class_type": "BasicScheduler", "inputs": {
        "model": [model_src, 0],
        "scheduler": "simple",
        "steps": int(steps),
        "denoise": 1.0,
    }}
    g["13"] = {"class_type": "RandomNoise", "inputs": {
        "noise_seed": int(seed),
    }}
    g["14"] = {"class_type": "SamplerCustomAdvanced", "inputs": {
        "noise": ["13", 0],
        "guider": ["10", 0],
        "sampler": ["11", 0],
        "sigmas": ["12", 0],
        "latent_image": ["5", 1],
    }}

    # 画面与音频解码
    g["902"] = {"class_type": "VAEDecode",
                "inputs": {"samples": ["14", 0], "vae": ["3", 0]}}
    g["903"] = {"class_type": "VAEDecodeAudio",
                "inputs": {"samples": ["14", 0], "vae": ["4", 0]}}
    g["904"] = {"class_type": "CreateVideo",
                "inputs": {"images": ["902", 0], "fps": 24.0, "audio": ["903", 0]}}
    g["905"] = {"class_type": "SaveVideo",
                "inputs": {"video": ["904", 0], "filename_prefix": "h3_ui",
                           "format": "auto", "codec": "auto"}}
    return g


# ---------------------------------------------------------------------------
# ComfyUI
# ---------------------------------------------------------------------------
def get_comfy_log(tail_chars=1200):
    for p in ["/content/comfy.log", "comfy.log"]:
        if os.path.exists(p):
            try:
                return open(p, "r", encoding="utf-8", errors="ignore").read()[-tail_chars:]
            except Exception:
                pass
    return None


def ensure_comfy_running():
    """检查 ComfyUI 是否存活，若未启动则尝试自动后台拉起"""
    try:
        if requests.get(f"{COMFY}/system_stats", timeout=2).status_code == 200:
            return True, "ComfyUI 运行正常"
    except Exception:
        pass

    main_py = COMFY_DIR / "main.py"
    if main_py.exists():
        try:
            import torch
            vram = torch.cuda.get_device_properties(0).total_memory / 2**30
            flag = "--highvram" if vram >= 70 else "--normalvram"
        except Exception:
            flag = "--normalvram"

        os.system(f"cd {COMFY_DIR} && nohup python main.py --listen 127.0.0.1 --port 8188 {flag} > /content/comfy.log 2>&1 &")
        for _ in range(25):
            time.sleep(2)
            try:
                if requests.get(f"{COMFY}/system_stats", timeout=2).status_code == 200:
                    return True, "ComfyUI 已自动拉起成功"
            except Exception:
                pass

    log_snip = get_comfy_log(1000)
    if log_snip:
        return False, f"ComfyUI (8188) 未响应。comfy.log 最新日志：\n{log_snip}"
    return False, "ComfyUI (8188) 未启动，请在 Colab 中先运行第 6 节启动 ComfyUI。"


def submit(graph):
    try:
        r = requests.post(f"{COMFY}/prompt",
                          json={"prompt": graph, "client_id": f"h3-{uuid.uuid4().hex[:8]}"},
                          timeout=30)
    except requests.exceptions.RequestException as e:
        log_snip = get_comfy_log(1000)
        if log_snip:
            raise RuntimeError(f"连接 ComfyUI (8188) 失败。comfy.log 最新日志：\n{log_snip}")
        raise RuntimeError(f"连接 ComfyUI (8188) 失败: {e}")

    if r.status_code != 200:
        raise RuntimeError(f"ComfyUI 拒收任务: {r.status_code} {r.text[:300]}")
    return r.json()["prompt_id"]


def fetch_video(history):
    for node_out in history.get("outputs", {}).values():
        for key in ("images", "videos", "gifs", "animated"):
            for item in node_out.get(key, []) or []:
                if not isinstance(item, dict):
                    continue
                fn = item.get("filename", "")
                if not fn.endswith((".mp4", ".webm", ".mov")):
                    continue
                url = (f"{COMFY}/view?filename={fn}"
                       f"&subfolder={item.get('subfolder', '')}"
                       f"&type={item.get('type', 'output')}")
                local = OUT_DIR / fn
                local.write_bytes(requests.get(url, timeout=300).content)
                try:
                    DRIVE_OUT.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(local, DRIVE_OUT / fn)
                    print(f"✅ 成片已成功同步到 Google Drive: {DRIVE_OUT / fn}")
                except Exception as exc:
                    print(f"⚠️ 同步到 Google Drive 失败: {exc}")
                return str(local)
    return None


# ---------------------------------------------------------------------------
# 界面动作
# ---------------------------------------------------------------------------
def rewrite_prompt(idea, key, model, seconds, ratio, first, last, style, extra):
    key = (key or GEMINI_KEY).strip()
    if not key:
        return "", "需要 Gemini API Key。去 aistudio.google.com/apikey 免费申请一个，粘到上面。"
    if not idea or not idea.strip():
        return "", "先写一句话描述你想要什么"

    length = snap_frames(int(seconds))
    duration = length / 24.0
    width, height = RATIOS.get(ratio, (1344, 768))
    mode = mode_of(first, last)

    req = [
        f"USER IDEA (Chinese, translate freely): {idea.strip()}",
        "",
        f"MODE: {mode}",
        f"DURATION: {duration:.2f} seconds ({length} frames at 24fps), aspect {width}x{height}",
        f"TARGET MARKET: Bilibili (Chinese audience)",
        f"STYLE REQUEST: {style or 'choose what fits the idea'}",
    ]
    if extra and extra.strip():
        req.append(f"EXTRA REQUIREMENTS: {extra.strip()}")
    if first:
        req.append("A first-frame image is provided: <Picture 1> is the exact 0.00s frame.")
    if last:
        req.append("A last-frame image is provided: it is the exact final frame of the video.")
    req += [
        "",
        "Write the full prompt now. It should be detailed and cinematic, "
        "covering every second of the timeline. Output only the prompt.",
    ]

    try:
        used = pick_model(key, model)
        text = call_gemini(key, used, "\n".join(req))
        return text, f"提示词已生成（{used}）· {mode} · {duration:.2f}s · {width}×{height}"
    except Exception as exc:
        return "", f"写提示词失败：{exc}"


def generate(idea, prompt, key, model, seconds, ratio, steps, seed, turbo,
             first, last, style, extra):
    """生成器：边跑边把进度推到界面上，避免看着像卡死。"""
    if not prompt or not prompt.strip():
        yield None, "还没有提示词。先点「帮我写提示词」。"
        return

    width, height = RATIOS.get(ratio, (1344, 768))
    length = snap_frames(int(seconds))
    if int(seed) < 0:
        seed = int.from_bytes(os.urandom(4), "big") % (2 ** 31)

    ok, msg = ensure_comfy_running()
    if not ok:
        yield None, f"出错：{msg}"
        return

    try:
        graph = build_graph(prompt.strip(), width, height, length,
                            int(steps), int(seed), bool(turbo), first, last)
        pid = submit(graph)
    except Exception as exc:
        yield None, f"出错：{exc}"
        return

    yield None, f"已提交 · {width}×{height} · {length} 帧 · 等待…"
    start = time.time()
    while True:
        time.sleep(3)
        try:
            hist = requests.get(f"{COMFY}/history/{pid}", timeout=10).json()
        except Exception:
            hist = {}
        if pid in hist:
            st = hist[pid].get("status", {})
            if st.get("status_str") == "error":
                yield None, f"生成失败：{str(st.get('messages'))[:400]}"
                return
            if st.get("status_str") == "success":
                break
        elapsed = int(time.time() - start)
        try:
            q = requests.get(f"{COMFY}/queue", timeout=5).json()
            pending = q.get("queue_pending", [])
            pos = next((i + 1 for i, p in enumerate(pending) if p[1] == pid), None)
            msg = (f"排队中，前面 {pos} 个 · 已等 {elapsed} 秒" if pos
                   else f"正在生成… 已用 {elapsed} 秒")
        except Exception:
            msg = f"生成中… {elapsed} 秒"
        yield None, msg

    video = fetch_video(hist)
    if not video:
        yield None, "跑完了但没找到视频文件"
        return
    yield video, f"完成 · {width}×{height} · {length} 帧 · seed {seed}"


def build_ui():
    with gr.Blocks(title="MiniMax H3") as app:
        gr.Markdown(
            "# MiniMax H3\n"
            "**写一句人话，剩下的交给 AI。** 它会按 H3 的官方规范把想法写成结构化提示词，再出片。"
        )

        with gr.Row():
            with gr.Column(scale=3):
                idea = gr.Textbox(
                    label="① 你想要什么效果（随便写，中文就行）",
                    lines=4,
                    placeholder="例：雨夜的赛博朋克街道，一个穿风衣的女孩背对镜头往前走，"
                                "霓虹灯倒映在积水里，镜头从地面缓慢升起到她的背影，"
                                "想要压抑的氛围，远处有警笛声。",
                )
            with gr.Column(scale=1):
                style = gr.Textbox(label="风格（可不填）", placeholder="如：黏土动画 / 3D CG / 黑白胶片")
                extra = gr.Textbox(label="额外要求（可不填）", placeholder="如：要一句台词 / 不要音乐")

        with gr.Row():
            seconds = gr.Dropdown([5, 6, 10], value=5, label="时长（秒）")
            ratio = gr.Dropdown(list(RATIOS.keys()), value="16:9 横屏", label="画幅")
            steps = gr.Slider(8, 30, value=20, step=1, label="步数")
            seed = gr.Number(value=-1, label="种子", precision=0)
            turbo = gr.Checkbox(value=False, label="加速模式")

        with gr.Accordion("首帧 / 尾帧（可选）", open=False):
            with gr.Row():
                first = gr.Image(type="filepath", label="首帧")
                last = gr.Image(type="filepath", label="尾帧")

        with gr.Accordion("Gemini API Key（写提示词用）", open=not bool(GEMINI_KEY)):
            key = gr.Textbox(label="API Key", type="password",
                             value=GEMINI_KEY,
                             info="aistudio.google.com/apikey 免费申请")
            model = gr.Textbox(label="模型（默认 Gemini 3.8 Flash + 高思考）", value="gemini-3.8-flash")

        gr.Markdown("---")

        rewrite_btn = gr.Button("② 帮我写提示词", variant="secondary")
        prompt = gr.Textbox(
            label="③ 生成的提示词（可以自己改，改完再生成）",
            lines=14,
            placeholder="点上面的按钮，AI 会把你的想法写成 H3 规范的长提示词…",
        )

        gen_btn = gr.Button("④ 生成视频", variant="primary", size="lg")
        video = gr.Video(label="结果")
        status = gr.Textbox(label="状态", interactive=False)

        rewrite_btn.click(
            rewrite_prompt,
            inputs=[idea, key, model, seconds, ratio, first, last, style, extra],
            outputs=[prompt, status],
        )
        gen_btn.click(
            generate,
            inputs=[idea, prompt, key, model, seconds, ratio, steps, seed, turbo,
                    first, last, style, extra],
            outputs=[video, status],
        )

        gr.Markdown(
            "---\n"
            "成片自动备份到 `Google Drive / h3-cloud / outputs`。\n\n"
            "**改了时长或首尾帧，要重新点一次「帮我写提示词」**——镜头时间轴是按参数写进去的。\n\n"
            "**慢或爆显存**：步数降到 12、时长 5 秒。"
        )
    return app


if __name__ == "__main__":
    print("正在检查并自动拉起 ComfyUI 后台引擎 (127.0.0.1:8188)...")
    ok, msg = ensure_comfy_running()
    print("ComfyUI 引擎状态:", msg)

    app = build_ui()
    app.queue()
    print("正在启动 Gradio 界面 (127.0.0.1:7860)...")
    print("Gemini key:", "已配置" if GEMINI_KEY else "未配置（可在界面填写）")
    app.launch(
        server_name="127.0.0.1",
        server_port=int(os.environ.get("H3_PORT", "7860")),
        share=True,
    )
