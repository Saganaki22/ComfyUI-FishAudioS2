<div align="center">
  <h1>🐟 ComfyUI-FishAudioS2</h1>

  <p>
    ComfyUI custom nodes for<br>
    <b><em>Fish Audio S2 Pro — Best TTS Among Open & Closed Source</em></b>
  </p>
  <p>
    <a href="https://fish.audio/"><img src="https://img.shields.io/badge/Playground-Fish_Audio-1f7a8c?style=flat-square&logo=readme&logoColor=white" alt="Fish Audio Playground"></a>
    <a href="https://huggingface.co/fishaudio/s2-pro"><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue' alt="HF Model"></a>
    <a href="https://huggingface.co/drbaph/s2-pro-fp8"><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Quantized-FP8-orange' alt="FP8 Model"></a>
    <a href="https://github.com/fishaudio/fish-speech"><img src="https://img.shields.io/badge/GitHub-Original-green" alt="GitHub"></a>
    <a href="https://huggingface.co/papers/2603.08823"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HF-Paper-yellow" alt="HF Paper"></a>
    <a href="https://arxiv.org/abs/2603.08823"><img src="https://img.shields.io/badge/arXiv-2603.08823-b31b1b" alt="arXiv"></a>
    <a href="https://discord.gg/Es5qTB9BcN"><img src="https://img.shields.io/discord/1214047546020728892?color=%23738ADB&label=Discord&logo=discord&logoColor=white&style=flat-square" alt="Discord"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-Fish%20Audio%20Research-yellow" alt="License"></a>
  </p>
</div>

---

<img width="1986" height="1242" alt="image" src="https://github.com/user-attachments/assets/d352ba24-2d52-4056-b61b-2ac2bb9ad00b" />


---

https://github.com/user-attachments/assets/d69377a6-1c28-40d0-a61a-ba27237e6801

---

## 🎵 Overview

**Fish Audio S2 Pro** is a state-of-the-art text-to-speech model with fine-grained inline control of prosody and emotion. Trained on 10M+ hours of audio data across **83 languages** with **1500+ emotive tags**, it combines reinforcement learning alignment with a Dual-Autoregressive architecture for speech that sounds natural, realistic, and emotionally rich.

**Paper:** [Fish Audio S2 Technical Report](https://arxiv.org/abs/2603.08823) (arXiv:2603.08823)

This ComfyUI wrapper provides native node-based integration with:
- **Zero-shot voice cloning** from 10-30 second reference audio
- **Inline emotion/prosody control** with `[tag]` syntax
- **Multi-speaker conversation synthesis** in a single pass
- **Per-speaker audio isolation** for multi-speaker lip sync workflows
- **83 language support** with automatic detection

---

## ✨ Features

- ** Zero-Shot Voice Cloning** – Clone any voice from 10-30 seconds of reference audio
- ** 1500+ Emotive Tags** – Fine-grained control with `[laugh]`, `[whisper]`, `[excited]`, `[sad]`, etc.
- ** 83 Languages** – Full multilingual support without phoneme preprocessing
- ** Multi-Speaker TTS** – Generate conversations with multiple cloned voices in one pass
- ** Per-Speaker Audio Isolation** – Separate audio tracks for each speaker (lip sync workflows)
- ** Native ComfyUI Integration** – AUDIO noodle inputs, progress bars, interruption support
- ** Optimized Performance** – Support for bf16/fp16/fp32 dtypes, SDPA, FlashAttention, SageAttention
- ** Smart Auto-Download** – Model weights auto-downloaded from HuggingFace on first use
- ** Smart Caching** – Optional model caching with automatic unloading on config change

---

##  Requirements

- **GPU:** NVIDIA GPU with **24GB+ VRAM** for full model (RTX 3090/4090, A5000, etc.)
  - **16GB+ VRAM** works with **BNB NF4 4-bit on-the-fly quantization** (~10-11 it/s)
  - **CPU/MPS:** ~1.5-2 seconds per token (experimental)
  - **18GB+ VRAM** works with **BNB INT8 on-the-fly quantization** (~10-11 it/s)
  - **20GB+ VRAM** works with the **FP8 quantized model** (`s2-pro-fp8`, ~15 it/s, requires RTX 4090/5090 or Ada/Blackwell GPU)
- **CPU/MPS:** ⚠️ EXPERIMENTAL
- **Python:** 3.10+
- **CUDA:** 11.8+ (for GPU inference)

> **⚠️ BNB On-the-Fly Quantization Requirements:**
> 
> BNB INT8 and BNB NF4 options use the **s2-pro (bf16)** model and quantize on-the-fly via bitsandbytes.
> 
> **Install bitsandbytes:**
> ```bash
> pip install bitsandbytes
> ```
> 
> **Note:** BNB options run at ~10-11 it/s vs ~15 it/s for FP8. They work on any NVIDIA GPU without special hardware requirements.

---

## Models

| Model | VRAM | Speed | Description |
|-------|------|-------|-------------|
| **s2-pro** | ~24GB | ~15-17 it/s | Full precision (4B params) — best quality, works out of the box. 15 it/s baseline, 17 it/s with SageAttention |
| **s2-pro-fp8** | ~20GB | ~15 it/s | FP8 weight-only quantized — **recommended for 20GB+ Ada/Blackwell GPUs** (RTX 4090/5090), no extra dependencies |
| **BNB INT8** | ~18GB | ~10-11 it/s | On-the-fly INT8 quantization via bitsandbytes — uses s2-pro model, requires bitsandbytes |
| **BNB NF4** | ~16GB | ~10-11 it/s | On-the-fly 4-bit NF4 quantization via bitsandbytes — uses s2-pro model, requires bitsandbytes |

Models are auto-downloaded from HuggingFace on first use:
- [fishaudio/s2-pro](https://huggingface.co/fishaudio/s2-pro) — full model
- [drbaph/s2-pro-fp8](https://huggingface.co/drbaph/s2-pro-fp8) — FP8 quantized

---

## Tested Configurations

**Tested and working v0.4.0 with PyTorch 2.10+cu13.**

| | Standalone env | Shared ComfyUI env | FP8 (RTX 4090/5090) |
|---|---|---|---|
| **Python** | 3.10 – 3.13 | 3.10 – 3.13 | 3.10 – 3.13 |
| **PyTorch** | 2.x + CUDA 11.8+ | managed by ComfyUI | 2.x + CUDA 11.8+ |
| **torchaudio** | any (2.9+ supported) | any (2.9+ supported) | any (2.9+ supported) |
| **protobuf** | any (not touched) | any (not touched) | any (not touched) |
| **descript-audio-codec** | 1.0.0 (`--no-deps`) | 1.0.0 (`--no-deps`) | 1.0.0 (`--no-deps`) |
| **descript-audiotools** | 0.7.2 (`--no-deps`) | 0.7.2 (`--no-deps`) | 0.7.2 (`--no-deps`) |
| **transformers** | ≥4.45.2 | ≥4.45.2 | ≥4.45.2 |
| **bitsandbytes** | optional (NF4/INT8) | optional (NF4/INT8) | not needed |
| **VRAM** | 24GB+ / 16GB+ (BNB) | 24GB+ / 16GB+ (BNB) | 20GB+ (Ada/Blackwell) |
| **GPU** | any NVIDIA | any NVIDIA | RTX 4090/5090 or Ada/Blackwell |

> As of v0.3.0, `descript-audio-codec`, `descript-audiotools`, and `protobuf` are never installed or modified by `pip install -r requirements.txt`. The two audio packages are auto-installed at first startup with `--no-deps`, leaving your environment's protobuf version untouched.
>
> As of v0.3.6, all transitive runtime dependencies of `dac`/`audiotools` (`flatten-dict`, `importlib-resources`, `julius`, `randomname`, `ffmpy`, `argbind`) are also auto-installed, fixing fresh-install failures on clean portable environments.

---

## Installation

<details>
<summary><b> Click to expand installation methods</b></summary>

### Method 1: ComfyUI Manager (Recommended)

1. Open ComfyUI Manager
2. Search for "FishAudioS2"
3. Click Install
4. Restart ComfyUI

### Method 2: Manual Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/saganaki22/ComfyUI-FishAudioS2.git
cd ComfyUI-FishAudioS2
pip install -r requirements.txt
```

> **Note:** `descript-audio-codec` and `descript-audiotools` are **not** in `requirements.txt` on purpose — they are auto-installed by the node at ComfyUI startup with `--no-deps` to avoid their `protobuf<5` constraint breaking other nodes in shared environments. You do not need to install them manually.
>
> If auto-install fails at startup, install them manually **with `--no-deps`** (omitting this flag can break other ComfyUI nodes that need protobuf 5.x):
> ```bash
> pip install descript-audio-codec --no-deps
> pip install "descript-audiotools>=0.7.2" --no-deps
> ```

</details>

> [!CAUTION]
> **Never run `pip install git+https://github.com/fishaudio/fish-speech`**
> fish-speech is already bundled inside this node. Running that command will downgrade PyTorch and other core packages, potentially breaking your entire ComfyUI environment. If you are seeing dependency errors at startup, restart ComfyUI once — the node auto-installs everything it needs.

---

## Quick Start

### Node Overview

| Node | Description |
|------|-------------|
| **Fish S2 TTS** | Text-to-speech with inline emotion tags |
| **Fish S2 Voice Clone TTS** | Voice cloning from reference audio + text |
| **Fish S2 Multi-Speaker TTS** | Multi-speaker conversation synthesis |
| **Fish S2 Multi-Speaker Split TTS** | Multi-speaker with per-speaker audio outputs |

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

## Node Reference

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
- `chunk_length`: Chunk length (100-400) [Will be removed in future update]
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
  > **Note:** [Whisper (MTB)](https://github.com/mel Mass/mtb_nodes) does **not** work with newer `transformers` versions and will throw an error. Manually write the reference text until the MTB maintainer merges the fix PR.

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
  > **Note:** [Whisper (MTB)](https://github.com/melMass/mtb_nodes) does **not** work with newer `transformers` versions and will throw an error. Manually write the reference text until the MTB maintainer merges the fix PR.

**Text Format:**
```
[speaker_1]: Hello, I'm speaker one.
[speaker_2]: And I'm speaker two!
```

**Outputs:**
- `audio`: Generated multi-speaker conversation (AUDIO)

---

### Fish S2 Multi-Speaker Split TTS

Multi-speaker conversation synthesis with separate audio tracks for each speaker. Perfect for multi-speaker lip sync workflows (e.g., Infinite Talk).

**Inputs:**
- Same as Fish S2 Multi-Speaker TTS

**Text Format:**
```
[speaker_1]: Hello, I'm speaker one.
[speaker_2]: And I'm speaker two!
```

**Outputs:**
- `audio`: Combined multi-speaker conversation (AUDIO)
- `speaker_1_audio` through `speaker_10_audio`: Isolated audio for each speaker (AUDIO)
  - Each speaker's track contains their speech when talking, silence otherwise
  - All tracks are the same length as the combined audio

> **Use Case:** Connect individual speaker outputs to separate lip sync nodes for multi-character animation.

> **Credit:** Per-speaker audio isolation idea suggested by [@lazybuttalented](https://github.com/lazybuttalented). This is an independent implementation.

---

## Emotive Tags

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

##  Supported Languages

**83 languages** supported without phoneme preprocessing:

**Tier 1 (Best Quality):** Japanese (ja), English (en), Chinese (zh)

**Tier 2:** Korean (ko), Spanish (es), Portuguese (pt), Arabic (ar), Russian (ru), French (fr), German (de)

**Full List:** sv, it, tr, no, nl, cy, eu, ca, da, gl, ta, hu, fi, pl, et, hi, la, ur, th, vi, jw, bn, yo, sl, cs, sw, nn, he, ms, uk, id, kk, bg, lv, my, tl, sk, ne, fa, af, el, bo, hr, ro, sn, mi, yi, am, be, km, is, az, sd, br, sq, ps, mn, ht, ml, sr, sa, te, ka, bs, pa, lt, kn, si, hy, mr, as, gu, fo

---

## File Structure

```
ComfyUI/
├── models/
│   └── fishaudioS2/
│       └── s2-pro/                    # Full model (auto-downloaded)
│           ├── model.pt
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
        ├── requirements.txt
        └── README.md
```

---

## Parameters Explained

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| **precision** | Model precision | `bfloat16` (CUDA), `float32` (CPU/MPS) |
| **attention** | Attention mechanism | `auto` (default), `sage_attention` (fastest, requires package) |
| **keep_model_loaded** | Cache model | `True` for multiple runs |
| **chunk_length**  | `200` (balanced), `100` (faster) |
| **temperature** | Sampling randomness | `0.7` (balanced), lower = more deterministic |
| **top_p** | Nucleus sampling | `0.7` (balanced) |
| **repetition_penalty** | Reduce repetition | `1.2` (default) |
| **compile_model** | torch.compile speedup | `True` (~10x after warmup, Linux only). Pin `max_new_tokens` to a fixed value when using compile — each new larger length triggers a recompile. |

---

## Troubleshooting

<details>
<summary><b>🛠️ Click to expand troubleshooting guide</b></summary>

### Models Not Downloading?

Manually download from [fishaudio/s2-pro](https://huggingface.co/fishaudio/s2-pro):
```bash
pip install -U huggingface_hub
huggingface-cli download fishaudio/s2-pro --local-dir ComfyUI/models/fishaudioS2/s2-pro
```

For the FP8 quantized model, download from [drbaph/s2-pro-fp8](https://huggingface.co/drbaph/s2-pro-fp8):
```bash
huggingface-cli download drbaph/s2-pro-fp8 --local-dir ComfyUI/models/fishaudioS2/s2-pro-fp8
```

### Protobuf Conflict in Shared Environments?

If you see errors like `ImportError: cannot import name 'runtime_version' from 'google.protobuf'` or dependency conflicts involving `descript-audiotools` / `descript-audio-codec` and `protobuf`, this is a known incompatibility between those packages' `protobuf<5` upper-bound and nodes that need protobuf 5.x (tensorflow, mediapipe, florence2, etc.).

As of v0.3.0 this is handled automatically — `descript-audio-codec` and `descript-audiotools` are installed at startup with `--no-deps` so their protobuf constraint is never enforced into the environment. Make sure you are on the latest version.

If you installed them manually before v0.3.0, reinstall with:
```bash
pip install descript-audio-codec --no-deps
pip install "descript-audiotools>=0.7.2" --no-deps
```

### Nodes Not Loading / Missing Dependencies?

All required packages are auto-installed on first startup. If the node fails to load, **restart ComfyUI once** — the installer runs before nodes register. If it still fails after a restart, install manually:

```bash
pip install -r requirements.txt
pip install flatten-dict importlib-resources julius randomname ffmpy argbind
pip install descript-audio-codec --no-deps
pip install "descript-audiotools>=0.7.2" --no-deps
```

Common missing packages:
- `sageattention` – for optimized attention (`pip install sageattention`)

> [!CAUTION]
> **Never run `pip install git+https://github.com/fishaudio/fish-speech`**
> fish-speech is bundled inside the node. Running that command will downgrade PyTorch and other packages, breaking your ComfyUI environment. Dependency errors on startup fix themselves after one restart.

### torchaudio / MockDecoder Error (PyTorch 2.9+)?

If you see `RuntimeError: Failed to create AudioDecoder ... MockDecoder() takes no arguments`, this is caused by torchaudio 2.9+ switching to the `torchcodec` backend which does not support in-memory audio buffers. Fixed in v0.3.6 — git pull to get the update.

### Conflicting `fish_speech` Package?

If you see `ImportError: cannot import name 'AUDIO_EXTENSIONS' from 'fish_speech_s2.utils.file'` pointing to another custom node's directory (e.g. `comfyui-mixlab-nodes`), a different node has its own `fish_speech` folder that conflicts with ours via `sys.path`. Disable the conflicting node or remove it. Do **not** pip-install `fish_speech` — it is bundled inside this node.

### Out of Memory?

- Use `bfloat16` precision instead of `float32`
- Set `keep_model_loaded=False`
- Reduce `chunk_length`
- Close other applications

### Slow Synthesis?

- Install SageAttention: `pip install sageattention`, then select `sage_attention`
- Enable `compile_model=True` (Linux only — pin `max_new_tokens` to avoid recompiles on varying lengths)
- Use GPU with CUDA support
- Enable `keep_model_loaded=True`

If errors persist, fall back to `sdpa` or `auto` attention.

</details>

---

## 🔗 Important Links

### 🤗 HuggingFace
- **Model (Full):** [fishaudio/s2-pro](https://huggingface.co/fishaudio/s2-pro)
- **Model (FP8 Quantized):** [drbaph/s2-pro-fp8](https://huggingface.co/drbaph/s2-pro-fp8)
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
