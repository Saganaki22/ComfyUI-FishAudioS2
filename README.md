<div align="center">
  <h1>🐟 ComfyUI-FishAudioS2</h1>

  <p>
    ComfyUI custom nodes for<br>
    <b><em>Fish Audio S2 Pro — Best TTS Among Open & Closed Source</em></b>
  </p>
  <p>
    <a href="https://fish.audio/"><img src="https://img.shields.io/badge/Playground-Fish_Audio-1f7a8c?style=flat-square&logo=readme&logoColor=white" alt="Fish Audio Playground"></a>
    <a href="https://huggingface.co/fishaudio/s2-pro"><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue' alt="HF Model"></a>
    <a href="https://huggingface.co/baicai1145/s2-pro-w4a16"><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Quantized-4bit-purple' alt="Quantized Model"></a>
    <a href="https://github.com/fishaudio/fish-speech"><img src="https://img.shields.io/badge/GitHub-Original-green" alt="GitHub"></a>
    <a href="https://huggingface.co/papers/2603.08823"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HF-Paper-yellow" alt="HF Paper"></a>
    <a href="https://arxiv.org/abs/2603.08823"><img src="https://img.shields.io/badge/arXiv-2603.08823-b31b1b" alt="arXiv"></a>
    <a href="https://discord.gg/Es5qTB9BcN"><img src="https://img.shields.io/discord/1214047546020728892?color=%23738ADB&label=Discord&logo=discord&logoColor=white&style=flat-square" alt="Discord"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-Fish%20Audio%20Research-yellow" alt="License"></a>
  </p>
</div>

---

<img width="2172" height="1225" alt="image" src="https://github.com/user-attachments/assets/d599d527-a0b0-4074-b958-47abbcedf98c" />

---

## 🎵 Overview

**Fish Audio S2 Pro** is a state-of-the-art text-to-speech model with fine-grained inline control of prosody and emotion. Trained on 10M+ hours of audio data across **83 languages** with **1500+ emotive tags**, it combines reinforcement learning alignment with a Dual-Autoregressive architecture for speech that sounds natural, realistic, and emotionally rich.

**Paper:** [Fish Audio S2 Technical Report](https://arxiv.org/abs/2603.08823) (arXiv:2603.08823)

This ComfyUI wrapper provides native node-based integration with:
- **Zero-shot voice cloning** from 10-30 second reference audio
- **Inline emotion/prosody control** with `[tag]` syntax
- **Multi-speaker conversation synthesis** in a single pass
- **83 language support** with automatic detection

---

## ✨ Features

- **🎤 Zero-Shot Voice Cloning** – Clone any voice from 10-30 seconds of reference audio
- **🎭 1500+ Emotive Tags** – Fine-grained control with `[laugh]`, `[whisper]`, `[excited]`, `[sad]`, etc.
- **🌍 83 Languages** – Full multilingual support without phoneme preprocessing
- **👥 Multi-Speaker TTS** – Generate conversations with multiple cloned voices in one pass
- **🔗 Native ComfyUI Integration** – AUDIO noodle inputs, progress bars, interruption support
- **⚡ Optimized Performance** – Support for bf16/fp16/fp32 dtypes, SDPA, FlashAttention, SageAttention
- **📦 Smart Auto-Download** – Model weights auto-downloaded from HuggingFace on first use
- **💾 Smart Caching** – Optional model caching with automatic unloading on config change

---

## 💻 Requirements

- **GPU:** NVIDIA GPU with **24GB+ VRAM** for full model (RTX 3090/4090, A5000, etc.)
  - **12GB+ VRAM** works with the **GPTQ W4A16 quantized model** (`s2-pro-w4a16`)
- **CPU/MPS:** Supported but significantly slower
- **Python:** 3.10+
- **CUDA:** 11.8+ (for GPU inference)

> **⚠️ GPTQ Quantized Model Requirements:**
> 
> The quantized model (`s2-pro-w4a16`) requires **AutoGPTQ** with CUDA kernels:
> 
> **Windows (Embedded Python) - Use prebuilt wheel:**
> ```cmd
> "path\to\ComfyUI\python_embeded\python.exe" -m pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu128/
> ```
> 
> **Linux - Build from bundled source:**
> ```bash
> cd ComfyUI/custom_nodes/ComfyUI-FishAudioS2/auto_gptq_src
> pip install -e .
> ```
> 
> **Linux - Or use prebuilt wheel:**
> ```bash
> pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu128/
> ```
> 
> Replace `cu128` with your CUDA version (`cu121`, `cu124`, etc.)

---

## 📦 Models

| Model | VRAM | Description |
|-------|------|-------------|
| **s2-pro** | ~24GB | Full precision (4B params) — best quality, works out of the box |
| **s2-pro-w4a16** | ~8GB | GPTQ 4-bit mixed precision — **recommended for 12GB GPUs**, requires AutoGPTQ |

The quantized model uses **GPTQ W4A16** (4-bit weights, 16-bit activations) for the Slow AR backbone while keeping the Fast AR decoder and codec in higher precision for quality.

Models are auto-downloaded from HuggingFace on first use:
- [fishaudio/s2-pro](https://huggingface.co/fishaudio/s2-pro) — full model
- [baicai1145/s2-pro-w4a16](https://huggingface.co/baicai1145/s2-pro-w4a16) — GPTQ quantized

---

## 📦 Installation

<details>
<summary><b>📥 Click to expand installation methods</b></summary>

### Method 1: ComfyUI Manager (Recommended)

1. Open ComfyUI Manager
2. Search for "FishAudioS2"
3. Click Install
4. Restart ComfyUI

### Method 2: Manual Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/YOUR_USERNAME/ComfyUI-FishAudioS2.git
cd ComfyUI-FishAudioS2
pip install -r requirements.txt
```

</details>

---

## 🚀 Quick Start

### Node Overview

| Node | Description |
|------|-------------|
| **Fish S2 TTS** | Text-to-speech with inline emotion tags |
| **Fish S2 Voice Clone TTS** | Voice cloning from reference audio + text |
| **Fish S2 Multi-Speaker TTS** | Multi-speaker conversation synthesis |

### Basic Workflow

1. **Download Model**
   - Models are auto-downloaded from [fishaudio/s2-pro](https://huggingface.co/fishaudio/s2-pro) on first use
   - Or manually download and place in `ComfyUI/models/fishaudioS2/`

2. **Text-to-Speech**
   - Add `Fish S2 TTS` node
   - Enter text with optional emotion tags: `Hello! [excited] This is Fish Audio S2.`
   - Select language (or use `auto`)
   - Run!

3. **Voice Cloning**
   - Add `Fish S2 Voice Clone TTS` node
   - Connect reference audio (10-30 seconds recommended)
   - Enter text to synthesize in cloned voice
   - Run!

4. **Multi-Speaker**
   - Add `Fish S2 Multi-Speaker TTS` node
   - Set number of speakers (2-10)
   - Connect reference audio for each speaker
   - Use `[speaker_1]:`, `[speaker_2]:` tokens in text
   - Run!

---

## 🎛️ Node Reference

### Fish S2 TTS

Text-to-speech synthesis with inline emotion/prosody control.

**Inputs:**
- `model_path`: S2-Pro checkpoint folder (place in `ComfyUI/models/fishaudioS2/`)
- `text`: Text to synthesize with optional `[tag]` emotion markers
- `language`: Language hint (`auto`, `en`, `zh`, `ja`, `ko`, etc.)
- `device`: Compute device (`auto`, `cuda`, `cpu`, `mps`)
- `precision`: Model precision (`bfloat16`, `float16`, `float32`)
- `attention`: Attention kernel (`auto`, `sdpa`, `sage_attention`, `flash_attention`)
- `max_new_tokens`: Maximum acoustic tokens (0 = auto)
- `chunk_length`: Chunk length for streaming (100-400)
- `temperature`: Sampling temperature (0.1-1.0)
- `top_p`: Top-p nucleus sampling (0.1-1.0)
- `repetition_penalty`: Repetition penalty (0.9-2.0)
- `seed`: Random seed
- `keep_model_loaded`: Cache model in VRAM between runs
- `compile_model`: Enable torch.compile (Linux only)

**Outputs:**
- `audio`: Generated speech (AUDIO)

---

### Fish S2 Voice Clone TTS

Voice cloning from reference audio.

**Inputs:**
- All inputs from Fish S2 TTS, plus:
- `reference_audio`: Reference audio to clone (10-30 seconds recommended)
- `reference_text` (optional): Transcript of reference audio for improved accuracy

**Outputs:**
- `audio`: Generated speech in cloned voice (AUDIO)

---

### Fish S2 Multi-Speaker TTS

Multi-speaker conversation synthesis.

**Inputs:**
- All inputs from Fish S2 TTS, plus:
- `num_speakers`: Number of speakers (2-10)
- `speaker_N_audio`: Reference audio for speaker N
- `speaker_N_ref_text`: Optional transcript for speaker N

**Text Format:**
```
[speaker_1]: Hello, I'm speaker one.
[speaker_2]: And I'm speaker two!
```

**Outputs:**
- `audio`: Generated multi-speaker conversation (AUDIO)

---

## 🎭 Emotive Tags

S2 Pro supports **1500+ unique emotive tags** using `[tag]` syntax. These are free-form natural language descriptions, not predefined tags.

**Common tags:**

| Category | Examples |
|----------|----------|
| **Emotion** | `[excited]`, `[sad]`, `[angry]`, `[surprised]`, `[delight]` |
| **Volume** | `[whisper]`, `[low voice]`, `[volume up]`, `[loud]`, `[shouting]`, `[screaming]` |
| **Pacing** | `[pause]`, `[short pause]`, `[inhale]`, `[exhale]`, `[sigh]` |
| **Vocalization** | `[laugh]`, `[laughing]`, `[chuckle]`, `[chuckling]`, `[tsk]`, `[clearing throat]` |
| **Tone** | `[professional broadcast tone]`, `[singing]`, `[with strong accent]` |
| **Expression** | `[moaning]`, `[panting]`, `[echo]`, `[pitch up]`, `[pitch down]` |

**Free-form examples:**
- `[whisper in small voice]`
- `[super happy and excited]`
- `[speaking slowly and clearly]`
- `[sarcastic tone]`

---

## 🌍 Supported Languages

**83 languages** supported without phoneme preprocessing:

**Tier 1 (Best Quality):** Japanese (ja), English (en), Chinese (zh)

**Tier 2:** Korean (ko), Spanish (es), Portuguese (pt), Arabic (ar), Russian (ru), French (fr), German (de)

**Full List:** sv, it, tr, no, nl, cy, eu, ca, da, gl, ta, hu, fi, pl, et, hi, la, ur, th, vi, jw, bn, yo, sl, cs, sw, nn, he, ms, uk, id, kk, bg, lv, my, tl, sk, ne, fa, af, el, bo, hr, ro, sn, mi, yi, am, be, km, is, az, sd, br, sq, ps, mn, ht, ml, sr, sa, te, ka, bs, pa, lt, kn, si, hy, mr, as, gu, fo

---

## 🗂️ File Structure

```
ComfyUI/
├── models/
│   └── fishaudioS2/
│       ├── s2-pro/                    # Full model (auto-downloaded)
│       │   ├── model.pt
│       │   └── config.json
│       └── s2-pro-w4a16/              # Quantized model (auto-downloaded)
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
        ├── fish_speech_src/           # Bundled fish-speech source
        ├── auto_gptq_src/             # Bundled AutoGPTQ source (for quantized model)
        ├── requirements.txt
        └── README.md
```

---

## 📊 Parameters Explained

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| **precision** | Model precision | `bfloat16` (CUDA), `float32` (CPU/MPS) |
| **attention** | Attention mechanism | `auto` (default), `sage_attention` (fastest, requires package) |
| **keep_model_loaded** | Cache model | `True` for multiple runs |
| **chunk_length**  | `200` (balanced), `100` (faster) |
| **temperature** | Sampling randomness | `0.7` (balanced), lower = more deterministic |
| **top_p** | Nucleus sampling | `0.7` (balanced) |
| **repetition_penalty** | Reduce repetition | `1.2` (default) |
| **compile_model** | torch.compile speedup | `True` (~10x after warmup) |

---

## 🔧 Troubleshooting

<details>
<summary><b>🛠️ Click to expand troubleshooting guide</b></summary>

### Models Not Downloading?

Manually download from [fishaudio/s2-pro](https://huggingface.co/fishaudio/s2-pro):
```bash
pip install -U huggingface_hub
huggingface-cli download fishaudio/s2-pro --local-dir ComfyUI/models/fishaudioS2/s2-pro
```

For the quantized model, download from [baicai1145/s2-pro-w4a16](https://huggingface.co/baicai1145/s2-pro-w4a16):
```bash
huggingface-cli download baicai1145/s2-pro-w4a16 --local-dir ComfyUI/models/fishaudioS2/s2-pro-w4a16
```

### GPTQ Quantized Model (s2-pro-w4a16)

The quantized model requires **AutoGPTQ**. Installation differs by platform:

<details>
<summary><b>Windows Instructions (Click to expand)</b></summary>

1. **Check your CUDA version:**
   ```cmd
   nvidia-smi
   ```
   Look for `CUDA Version: 12.x`

2. **Install AutoGPTQ** using embedded Python (adjust path and CUDA version):

   **For ComfyUI Portable (CUDA 12.8):**
   ```cmd
   "python_embeded\python.exe" -m pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu128/
   ```

   **For ComfyUI Portable (CUDA 12.4):**
   ```cmd
   "python_embeded\python.exe" -m pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu124/
   ```

   > **Note:** Adjust the path to match your actual ComfyUI installation location.

3. **Restart ComfyUI**

</details>

<details>
<summary><b>Linux Instructions (Click to expand)</b></summary>

**Option 1: Prebuilt Wheel (Recommended)**
```bash
pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu121/
```

**Option 2: Build from Bundled Source**
```bash
cd ComfyUI/custom_nodes/ComfyUI-FishAudioS2/auto_gptq_src
pip install -e .
```

</details>

> **If installation fails:** Use the full model (`s2-pro`) instead - it works on all systems with 24GB+ VRAM.

### Missing Dependencies?

Install all dependencies:
```bash
cd ComfyUI/custom_nodes/ComfyUI-FishAudioS2
pip install -r requirements.txt
```

Common missing packages:
- `sageattention` – for optimized attention (`pip install sageattention`)

### Out of Memory?

- Use `bfloat16` precision instead of `float32`
- Set `keep_model_loaded=False`
- Reduce `chunk_length`
- Close other applications

### Slow Synthesis?

- Install SageAttention: `pip install sageattention`, then select `sage_attention`
- Enable `compile_model=True`
- Use GPU with CUDA support
- Enable `keep_model_loaded=True`

If errors persist, fall back to `sdpa` or `auto` attention.

### GPTQ Quantized Model Not Working?

The GPTQ model (`s2-pro-w4a16`) requires AutoGPTQ with CUDA kernels.

**Windows (Portable/Embedded Python):**
1. Check CUDA version: `nvidia-smi`
2. Install with prebuilt wheel:
   ```cmd
   "path\to\ComfyUI\python_embeded\python.exe" -m pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu128/
   ```
   Replace `cu128` with your CUDA version (`cu121`, `cu124`, `cu128`)

**Linux:**
```bash
cd ComfyUI/custom_nodes/ComfyUI-FishAudioS2/auto_gptq_src
pip install -e .
```

Or use prebuilt wheels:
```bash
pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu121/
```

**If all else fails:** Use the full model (`s2-pro`) which works on all systems with 24GB+ VRAM.

</details>

---

## 🔗 Important Links

### 🤗 HuggingFace
- **Model (Full):** [fishaudio/s2-pro](https://huggingface.co/fishaudio/s2-pro)
- **Model (4-bit Quantized):** [baicai1145/s2-pro-w4a16](https://huggingface.co/baicai1145/s2-pro-w4a16)
- **Paper:** [huggingface.co/papers/2603.08823](https://huggingface.co/papers/2603.08823)

### 📄 Paper & Code
- **arXiv Paper:** [arxiv.org/abs/2603.08823](https://arxiv.org/abs/2603.08823)
- **Official Repository:** [fishaudio/fish-speech](https://github.com/fishaudio/fish-speech)
- **Documentation:** [speech.fish.audio](https://speech.fish.audio/)

### 🌐 Community
- **Playground:** [fish.audio](https://fish.audio/)
- **Discord:** [Fish Audio Discord](https://discord.gg/Es5qTB9BcN)
- **Blog:** [Fish Audio S2 Release](https://fish.audio/blog/fish-audio-open-sources-s2/)

---

## 📄 Citation

If you use Fish Audio S2 in your research, please cite:

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

## 📄 License

This project uses the [Fish Audio Research License](LICENSE). Research and non-commercial use is permitted. Commercial use requires a separate license from Fish Audio — contact business@fish.audio.

Model weights from [fishaudio/s2-pro](https://huggingface.co/fishaudio/s2-pro) are subject to the same license.

## ⚠️ Usage Disclaimer

Fish Audio S2 is intended for academic research, educational purposes, and legitimate applications. Please use responsibly and ethically. We do not hold any responsibility for any illegal usage. Please refer to your local laws about DMCA and related regulations.

---

<div align="center">
    <b><em>Best-in-class TTS with Voice Cloning for ComfyUI</em></b>
</div>
