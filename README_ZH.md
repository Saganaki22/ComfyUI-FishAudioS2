<div align="center">
  <h1>🐟 ComfyUI-FishAudioS2</h1>

  <p>
    ComfyUI 自定义节点<br>
    <b><em>Fish Audio S2 Pro — 开源与闭源中最优秀的 TTS</em></b>
  </p>
  <p>
    <a href="https://fish.audio/"><img src="https://img.shields.io/badge/Playground-Fish_Audio-1f7a8c?style=flat-square&logo=readme&logoColor=white" alt="Fish Audio Playground"></a>
    <a href="https://huggingface.co/fishaudio/s2-pro"><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue' alt="HF Model"></a>
    <a href="https://huggingface.co/baicai1145/s2-pro-w4a16"><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Quantized-4bit-purple' alt="Quantized Model"></a>
    <a href="https://huggingface.co/drbaph/s2-pro-fp8"><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Quantized-FP8-orange' alt="FP8 Model"></a>
    <a href="https://github.com/fishaudio/fish-speech"><img src="https://img.shields.io/badge/GitHub-Original-green" alt="GitHub"></a>
    <a href="https://huggingface.co/papers/2603.08823"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HF-Paper-yellow" alt="HF Paper"></a>
    <a href="https://arxiv.org/abs/2603.08823"><img src="https://img.shields.io/badge/arXiv-2603.08823-b31b1b" alt="arXiv"></a>
    <a href="https://discord.gg/Es5qTB9BcN"><img src="https://img.shields.io/discord/1214047546020728892?color=%23738ADB&label=Discord&logo=discord&logoColor=white&style=flat-square" alt="Discord"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-Fish%20Audio%20Research-yellow" alt="License"></a>
  </p>
</div>

---

## 🎵 概述

**Fish Audio S2 Pro** 是最先进的文本转语音模型，具有细粒度的韵律和情感控制。在 **83 种语言**、超过 **1000 万小时**的音频数据上训练，支持 **1500+ 情感标签**，结合强化学习对齐和双自回归架构，生成自然、逼真、情感丰富的语音。

**论文：** [Fish Audio S2 技术报告](https://arxiv.org/abs/2603.08823) (arXiv:2603.08823)

本 ComfyUI 封装提供原生节点集成：
- **零样本声音克隆** - 仅需 10-30 秒参考音频
- **内联情感/韵律控制** - 使用 `[标签]` 语法
- **多说话人对话合成** - 单次生成多角色对话
- **83 种语言支持** - 自动检测语言

---

## ✨ 特性

- **🎤 零样本声音克隆** – 仅需 10-30 秒参考音频即可克隆任意声音
- **🎭 1500+ 情感标签** – 细粒度控制：`[laugh]`、`[whisper]`、`[excited]`、`[sad]` 等
- **🌍 83 种语言** – 无需音素预处理的全多语言支持
- **👥 多说话人 TTS** – 一次生成多个克隆声音的对话
- **🔗 原生 ComfyUI 集成** – AUDIO 连接输入、进度条、中断支持
- **⚡ 性能优化** – 支持 bf16/fp16/fp32、SDPA、FlashAttention、SageAttention
- **📦 智能自动下载** – 首次使用时自动从 HuggingFace 下载模型权重
- **💾 智能缓存** – 可选模型缓存，配置变更时自动卸载

---

## 💻 系统要求

- **GPU：** NVIDIA 显卡，完整模型需 **24GB+ 显存**（RTX 3090/4090、A5000 等）
  - **12GB+ 显存** 可使用 **GPTQ W4A16 量化模型**（`s2-pro-w4a16`）
  - **12GB+ 显存** 可使用 **FP8 量化模型**（`s2-pro-fp8`，需要 RTX 4090/5090 或任何支持 FP8 的 Ada/Blackwell 显卡）
- **CPU/MPS：** 支持但速度明显较慢
- **Python：** 3.10+
- **CUDA：** 11.8+（GPU 推理）

> **⚠️ GPTQ 量化模型要求：**
> 
> 量化模型（`s2-pro-w4a16`）需要带 CUDA 内核的 **AutoGPTQ**：
> 
> **Windows（嵌入式 Python）- 使用预编译包：**
> ```cmd
> "path\to\ComfyUI\python_embeded\python.exe" -m pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu128/
> ```
> 
> **Linux - 从捆绑源码编译：**
> ```bash
> cd ComfyUI/custom_nodes/ComfyUI-FishAudioS2/auto_gptq_src
> pip install -e .
> ```
> 
> **Linux - 或使用预编译包：**
> ```bash
> pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu128/
> ```
> 
> 将 `cu128` 替换为您的 CUDA 版本（`cu121`、`cu124` 等）

---

## 📦 模型

| 模型 | 显存 | 描述 |
|------|------|------|
| **s2-pro** | ~24GB | 完整精度（40亿参数）— 最佳质量，开箱即用 |
| **s2-pro-w4a16** | ~8GB | GPTQ 4位混合精度 — **12GB 显卡推荐**，需要 AutoGPTQ |
| **s2-pro-fp8** | ~12GB | FP8 逐行缩放量化 — **推荐用于 Ada/Blackwell 显卡**（RTX 4090/5090），无需额外依赖 |

首次使用时自动从 HuggingFace 下载模型：
- [fishaudio/s2-pro](https://huggingface.co/fishaudio/s2-pro) — 完整模型
- [baicai1145/s2-pro-w4a16](https://huggingface.co/baicai1145/s2-pro-w4a16) — GPTQ 4位量化
- [drbaph/s2-pro-fp8](https://huggingface.co/drbaph/s2-pro-fp8) — FP8 量化

---

## 📦 安装

<details>
<summary><b>📥 点击展开安装方法</b></summary>

### 方法 1：ComfyUI Manager（推荐）

1. 打开 ComfyUI Manager
2. 搜索 "FishAudioS2"
3. 点击安装
4. 重启 ComfyUI

### 方法 2：手动安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Saganaki22/ComfyUI-FishAudioS2.git
cd ComfyUI-FishAudioS2
pip install -r requirements.txt
```

</details>

---

## 🚀 快速开始

### 节点概览

| 节点 | 描述 |
|------|------|
| **Fish S2 TTS** | 带内联情感标签的文本转语音 |
| **Fish S2 Voice Clone TTS** | 从参考音频克隆声音 |
| **Fish S2 Multi-Speaker TTS** | 多说话人对话合成 |

### 基础工作流

1. **下载模型**
   - 首次使用时自动从 [fishaudio/s2-pro](https://huggingface.co/fishaudio/s2-pro) 下载
   - 或手动下载放置到 `ComfyUI/models/fishaudioS2/`

2. **文本转语音**
   - 添加 `Fish S2 TTS` 节点
   - 输入带可选情感标签的文本：`你好！[excited] 这是 Fish Audio S2。`
   - 选择语言（或使用 `auto`）
   - 运行！

3. **声音克隆**
   - 添加 `Fish S2 Voice Clone TTS` 节点
   - 连接参考音频（推荐 10-30 秒）
   - 输入要合成的文本
   - 运行！

4. **多说话人**
   - 添加 `Fish S2 Multi-Speaker TTS` 节点
   - 设置说话人数量（2-10）
   - 为每个说话人连接参考音频
   - 在文本中使用 `[speaker_1]:`、`[speaker_2]:` 标签
   - 运行！

---

## 🎛️ 节点参考

### Fish S2 TTS

带内联情感/韵律控制的文本转语音合成。

**输入：**
- `model_path`：S2-Pro 检查点文件夹（放置于 `ComfyUI/models/fishaudioS2/`）
- `text`：要合成的文本，可带 `[标签]` 情感标记
- `language`：语言提示（`auto`、`en`、`zh`、`ja`、`ko` 等）
- `device`：计算设备（`auto`、`cuda`、`cpu`、`mps`）
- `precision`：模型精度（`auto`、`bfloat16`、`float16`、`float32`）
- `attention`：注意力内核（`auto`、`sdpa`、`sage_attention`、`flash_attention`）
- `max_new_tokens`：最大声学 token 数（0 = 自动）
- `chunk_length`：流式分块长度（100-400）
- `temperature`：采样温度（0.1-1.0）
- `top_p`：Top-p 核采样（0.1-1.0）
- `repetition_penalty`：重复惩罚（0.9-2.0）
- `seed`：随机种子
- `keep_model_loaded`：在显存中缓存模型
- `compile_model`：启用 torch.compile（仅 Linux）

**输出：**
- `audio`：生成的语音（AUDIO）

---

### Fish S2 Voice Clone TTS

从参考音频克隆声音。

**输入：**
- Fish S2 TTS 的所有输入，加上：
- `reference_audio`：要克隆的参考音频（推荐 10-30 秒）
- `reference_text`（可选）：参考音频的文字稿，提高准确性

**输出：**
- `audio`：以克隆声音生成的语音（AUDIO）

---

### Fish S2 Multi-Speaker TTS

多说话人对话合成。

**输入：**
- Fish S2 TTS 的所有输入，加上：
- `num_speakers`：说话人数量（2-10）
- `speaker_N_audio`：说话人 N 的参考音频
- `speaker_N_ref_text`：说话人 N 的可选文字稿
- `pause_after_speaker`：说话人间隔时间（默认 0.4 秒）

**文本格式：**
```
[speaker_1]:你好，我是说话人一。
[speaker_2]:我是说话人二！
[speaker_1]:[laugh] 很高兴认识你。
```

**输出：**
- `audio`：生成的多说话人对话（AUDIO）

---

## 🎭 情感标签

S2 Pro 使用 `[标签]` 语法支持 **1500+ 种独特情感标签**。这些是自由形式的自然语言描述，而非预定义标签。

**常用标签：**

| 类别 | 示例 |
|------|------|
| **情感** | `[excited]`、`[sad]`、`[angry]`、`[surprised]`、`[delight]` |
| **音量** | `[whisper]`、`[low voice]`、`[volume up]`、`[loud]`、`[shouting]`、`[screaming]` |
| **节奏** | `[pause]`、`[short pause]`、`[inhale]`、`[exhale]`、`[sigh]` |
| **发声** | `[laugh]`、`[laughing]`、`[chuckle]`、`[chuckling]`、`[tsk]`、`[clearing throat]` |
| **语调** | `[professional broadcast tone]`、`[singing]`、`[with strong accent]` |
| **表达** | `[moaning]`、`[panting]`、`[echo]`、`[pitch up]`、`[pitch down]` |

**自由形式示例：**
- `[whisper in small voice]`
- `[super happy and excited]`
- `[speaking slowly and clearly]`
- `[sarcastic tone]`

---

## 🌍 支持的语言

无需音素预处理，支持 **83 种语言**：

**第一梯队（最佳质量）：** 日语（ja）、英语（en）、中文（zh）

**第二梯队：** 韩语（ko）、西班牙语（es）、葡萄牙语（pt）、阿拉伯语（ar）、俄语（ru）、法语（fr）、德语（de）

**完整列表：** sv, it, tr, no, nl, cy, eu, ca, da, gl, ta, hu, fi, pl, et, hi, la, ur, th, vi, jw, bn, yo, sl, cs, sw, nn, he, ms, uk, id, kk, bg, lv, my, tl, sk, ne, fa, af, el, bo, hr, ro, sn, mi, yi, am, be, km, is, az, sd, br, sq, ps, mn, ht, ml, sr, sa, te, ka, bs, pa, lt, kn, si, hy, mr, as, gu, fo

---

## 🗂️ 文件结构

```
ComfyUI/
├── models/
│   └── fishaudioS2/
│       ├── s2-pro/                    # 完整模型（自动下载）
│       │   ├── model.pt
│       │   └── config.json
│       └── s2-pro-w4a16/              # 量化模型（自动下载）
│           ├── model.safetensors
│           └── config.json
└── custom_nodes/
    └── ComfyUI-FishAudioS2/
        ├── __init__.py
        ├── nodes/
        │   ├── tts_node.py
        │   ├── voice_clone_node.py
        │   ├── multi_speaker_node.py
        │   ├── loader.py
        │   └── model_cache.py
        ├── fish_speech_src/           # 捆绑的 fish-speech 源码
        ├── auto_gptq_src/             # 捆绑的 AutoGPTQ 源码（用于量化模型）
        ├── requirements.txt
        └── README.md
```

---

## 📊 参数说明

| 参数 | 描述 | 推荐值 |
|------|------|--------|
| **precision** | 模型精度 | `auto`（自动），`bfloat16`（CUDA），`float32`（CPU/MPS） |
| **attention** | 注意力机制 | `auto`（默认），`sage_attention`（最快，需安装） |
| **keep_model_loaded** | 缓存模型 | 多次运行时设为 `True` |
| **chunk_length** | 分块长度 | `200`（平衡），`100`（更快） |
| **temperature** | 采样随机性 | `0.7`（平衡），越低越确定 |
| **top_p** | 核采样 | `0.7`（平衡） |
| **repetition_penalty** | 减少重复 | `1.2`（默认） |
| **compile_model** | torch.compile 加速 | `True`（预热后约 10 倍） |

---

## 🔧 故障排除

<details>
<summary><b>🛠️ 点击展开故障排除指南</b></summary>

### 模型无法下载？

从 [fishaudio/s2-pro](https://huggingface.co/fishaudio/s2-pro) 手动下载：
```bash
pip install -U huggingface_hub
huggingface-cli download fishaudio/s2-pro --local-dir ComfyUI/models/fishaudioS2/s2-pro
```

GPTQ 量化模型从 [baicai1145/s2-pro-w4a16](https://huggingface.co/baicai1145/s2-pro-w4a16) 下载：
```bash
huggingface-cli download baicai1145/s2-pro-w4a16 --local-dir ComfyUI/models/fishaudioS2/s2-pro-w4a16
```

FP8 量化模型从 [drbaph/s2-pro-fp8](https://huggingface.co/drbaph/s2-pro-fp8) 下载：
```bash
huggingface-cli download drbaph/s2-pro-fp8 --local-dir ComfyUI/models/fishaudioS2/s2-pro-fp8
```

### GPTQ 量化模型（s2-pro-w4a16）

量化模型需要 **AutoGPTQ**。不同平台安装方式不同：

<details>
<summary><b>Windows 说明（点击展开）</b></summary>

1. **检查 CUDA 版本：**
   ```cmd
   nvidia-smi
   ```
   查看 `CUDA Version: 12.x`

2. **使用嵌入式 Python 安装 AutoGPTQ**（调整路径和 CUDA 版本）：

   **ComfyUI 便携版（CUDA 12.8）：**
   ```cmd
   "python_embeded\python.exe" -m pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu128/
   ```

   **ComfyUI 便携版（CUDA 12.4）：**
   ```cmd
   "python_embeded\python.exe" -m pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu124/
   ```

   > **注意：** 调整路径以匹配您的实际 ComfyUI 安装位置。

3. **重启 ComfyUI**

</details>

<details>
<summary><b>Linux 说明（点击展开）</b></summary>

**方法 1：预编译包（推荐）**
```bash
pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu121/
```

**方法 2：从捆绑源码编译**
```bash
cd ComfyUI/custom_nodes/ComfyUI-FishAudioS2/auto_gptq_src
pip install -e .
```

</details>

> **如果安装失败：** 改用完整模型（`s2-pro`）— 在所有 24GB+ 显存系统上均可使用。

### 缺少依赖？

安装所有依赖：
```bash
cd ComfyUI/custom_nodes/ComfyUI-FishAudioS2
pip install -r requirements.txt
```

常见缺少的包：
- `sageattention` – 用于优化注意力（`pip install sageattention`）

### 显存不足？

- 使用 `bfloat16` 精度而非 `float32`
- 设置 `keep_model_loaded=False`
- 减少 `chunk_length`
- 关闭其他应用程序

### 合成速度慢？

- 安装 SageAttention：`pip install sageattention`，然后选择 `sage_attention`
- 启用 `compile_model=True`
- 使用支持 CUDA 的 GPU
- 启用 `keep_model_loaded=True`

如果仍有错误，回退到 `sdpa` 或 `auto` 注意力。

### GPTQ 量化模型不工作？

GPTQ 模型（`s2-pro-w4a16`）需要带 CUDA 内核的 AutoGPTQ。

**Windows（便携版/嵌入式 Python）：**
1. 检查 CUDA 版本：`nvidia-smi`
2. 使用预编译包安装：
   ```cmd
   "path\to\ComfyUI\python_embeded\python.exe" -m pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu128/
   ```
   将 `cu128` 替换为您的 CUDA 版本（`cu121`、`cu124`、`cu128`）

**Linux：**
```bash
cd ComfyUI/custom_nodes/ComfyUI-FishAudioS2/auto_gptq_src
pip install -e .
```

或使用预编译包：
```bash
pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu121/
```

**如果以上都失败：** 使用完整模型（`s2-pro`），在所有 24GB+ 显存系统上均可使用。

</details>

---

## 🔗 重要链接

### 🤗 HuggingFace
- **模型（完整）：** [fishaudio/s2-pro](https://huggingface.co/fishaudio/s2-pro)
- **模型（4位量化）：** [baicai1145/s2-pro-w4a16](https://huggingface.co/baicai1145/s2-pro-w4a16)
- **模型（FP8 量化）：** [drbaph/s2-pro-fp8](https://huggingface.co/drbaph/s2-pro-fp8)
- **论文：** [huggingface.co/papers/2603.08823](https://huggingface.co/papers/2603.08823)

### 📄 论文与代码
- **arXiv 论文：** [arxiv.org/abs/2603.08823](https://arxiv.org/abs/2603.08823)
- **官方仓库：** [fishaudio/fish-speech](https://github.com/fishaudio/fish-speech)
- **文档：** [speech.fish.audio](https://speech.fish.audio/)

### 🌐 社区
- **在线体验：** [fish.audio](https://fish.audio/)
- **Discord：** [Fish Audio Discord](https://discord.gg/Es5qTB9BcN)
- **博客：** [Fish Audio S2 发布](https://fish.audio/blog/fish-audio-open-sources-s2/)

---

## 📄 引用

如果您在研究中使用 Fish Audio S2，请引用：

```bibtex
@misc{liao2026fishaudios2technical,
      title={Fish Audio S2 Technical Report}, 
      author={Shijia Liao and Yuxuan Wang and Songting Liu and Yifan Cheng and Ruoyi Zhang and Tianyu Li and Shidong Li and Yisheng Zheng and Xingwei Liu and Qingzheng Wang and Zhizhuo Zhou and Jiahua Liu and Xin Chen and Dawei Han},
      year={2026},
      eprint={2603.08823},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2603.08823},
}
```

---

## 📄 许可证

本项目使用 [Fish Audio 研究许可证](LICENSE)。允许研究和非商业用途。商业用途需从 Fish Audio 获取单独许可证 — 联系 business@fish.audio。

来自 [fishaudio/s2-pro](https://huggingface.co/fishaudio/s2-pro) 的模型权重受相同许可证约束。

## ⚠️ 使用声明

Fish Audio S2 旨在用于学术研究、教育目的和合法应用。请负责任地、合乎道德地使用。我们不对任何非法使用承担责任。请参阅您当地关于 DMCA 和相关法规的法律。

---

<div align="center">
    <b><em>ComfyUI 中最优秀的带声音克隆 TTS</em></b>
</div>
