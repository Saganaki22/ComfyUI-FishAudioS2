"""ComfyUI custom nodes for Fish Audio S2-Pro TTS.

Provides four nodes:
  - FishS2TTS             — text → speech, 80+ languages, inline emotion tags
  - FishS2VoiceCloneTTS   — reference audio + text → cloned-voice speech
  - FishS2MultiSpeakerTTS — multi-speaker conversation synthesis in one pass
  - FishS2MultiSpeakerSplitTTS — multi-speaker with per-speaker audio outputs

Required pip packages are auto-installed on startup.
Model weights are auto-downloaded from HuggingFace on first inference.
"""

__version__ = "0.4.4"

import importlib
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

# ---------------------------------------------------------------------------
# Bundle fish-speech source — add it to sys.path so `import fish_speech` and
# `import tools` resolve from our bundled copy, not from pip (which can't
# install it reliably into embedded Python).
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent.resolve()
_FISH_SRC = _HERE / "fish_speech_src"
# Add fish_speech_src to path
if _FISH_SRC.is_dir():
    _fish_src_str = str(_FISH_SRC)
    if _fish_src_str not in sys.path:
        sys.path.insert(0, _fish_src_str)

logger = logging.getLogger("FishAudioS2")
logger.propagate = False

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[FishAudioS2] %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# pip helper — works for portable embedded Python, venv, conda, system Python
# ---------------------------------------------------------------------------

def _find_pip() -> list[str]:
    """
    Return the pip/uv command that installs into the same environment as the
    currently-running Python — regardless of install type.

    Tries python -m pip first; if pip is not installed (common in uv-managed
    venvs), falls back to python -m uv pip, then standalone uv pip.
    """
    import shutil

    embedded = "python_embeded" in sys.executable
    base = [sys.executable] + (["-s"] if embedded else [])

    # 1. try pip
    try:
        subprocess.check_output(
            base + ["-m", "pip", "--version"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return base + ["-m", "pip"]
    except Exception:
        pass

    # 2. try uv as python module
    try:
        subprocess.check_output(
            base + ["-m", "uv", "--version"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        logger.info("Using uv (python module) for package installs.")
        return base + ["-m", "uv", "pip"]
    except Exception:
        pass

    # 3. try standalone uv on PATH
    if shutil.which("uv"):
        logger.info("Using standalone uv for package installs.")
        return ["uv", "pip"]

    # 4. fall back to pip (will fail with a clear error)
    return base + ["-m", "pip"]


def _pip_install(spec: str) -> bool:
    """
    Install a package. spec may include flags like '--no-deps'.
    Splits on whitespace so flags are passed as separate args to pip.
    Returns True on success.
    """
    cmd = _find_pip() + ["install"] + spec.split()
    logger.info(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
        )
        if result.returncode == 0:
            logger.info(f"Successfully installed: {spec}")
            # Invalidate Python's import-system filesystem cache so the newly
            # installed package is visible to __import__ without a restart.
            importlib.invalidate_caches()
            return True
        logger.error(f"pip install failed for '{spec}':\n{result.stderr.strip()}")
        return False
    except subprocess.TimeoutExpired:
        logger.error(f"pip install timed out for: {spec}")
        return False
    except Exception as e:
        logger.error(f"pip install error for '{spec}': {e}")
        return False


# Packages to auto-install if missing: (import_name, pip_install_spec)
# fish_speech is bundled in fish_speech_src/ — NOT installed via pip.
# Only its runtime deps are installed here. torch is intentionally excluded.
_REQUIRED = [
    ("numpy",           "numpy"),
    ("tqdm",            "tqdm"),
    ("soundfile",       "soundfile"),
    ("loguru",          "loguru"),
    ("transformers",    "transformers>=4.45.2"),
    ("einops",          "einops>=0.7.0"),
    ("librosa",         "librosa>=0.10.1"),
    ("rich",            "rich>=13.5.3"),
    ("ormsgpack",       "ormsgpack"),
    ("pydantic",        "pydantic==2.9.2"),
    ("tiktoken",        "tiktoken>=0.8.0"),
    ("cachetools",      "cachetools"),
    ("zstandard",       "zstandard>=0.22.0"),
    ("resampy",         "resampy>=0.4.3"),
    ("safetensors",     "safetensors>=0.4.0"),
    ("pyrootutils",     "pyrootutils>=1.0.4"),
    ("natsort",         "natsort>=8.4.0"),
    ("loralib",         "loralib>=0.1.2"),
    ("hydra",           "hydra-core>=1.3.2"),
    # einx is an optional dep used only by certain attention backends.
    # It is NOT hard-required for TTS inference — skip if it can't import
    # (e.g. jax namespace conflicts in some environments).
    # ("einx",          "einx==0.2.2"),
    # These are direct runtime imports of dac/audiotools (not training deps).
    # Must be installed before dac/audiotools even with --no-deps.
    #   flatten_dict:        audiotools/core/util.py
    #   importlib_resources: audiotools/core/playback.py
    #   julius:              audiotools/core/dsp.py, loudness.py
    #   randomname:          audiotools/ml/experiment.py
    #   ffmpy:               audiotools/core/ffmpeg.py (via audio_signal.py FFMPEGMixin)
    #   argbind:             dac/utils/__init__.py (imported by dac/__init__.py)
    #   tensorboard:         audiotools/ml/__init__.py (imported at module load)
    ("flatten_dict",        "flatten-dict"),
    ("importlib_resources", "importlib-resources"),
    ("julius",              "julius"),
    ("randomname",          "randomname"),
    ("ffmpy",               "ffmpy"),
    ("argbind",             "argbind"),
    ("tensorboard",         "tensorboard"),
    # Install dac/audiotools with --no-deps to avoid their protobuf<5 upper-bound
    # constraint being enforced into the environment. All of their runtime deps
    # that matter for inference (numpy, torch, einops, etc.) are already covered
    # by entries above. protobuf is NOT needed for TTS inference — it is only
    # used by fish-speech's training dataset tooling which is never called here.
    ("dac",             "descript-audio-codec --no-deps"),
    ("audiotools",      "descript-audiotools>=0.7.2 --no-deps"),
    ("bitsandbytes",    "bitsandbytes"),
]


def _ensure_fish_source() -> bool:
    """
    Add the bundled fish_speech_src/ to sys.path and verify it is importable.
    The source is shipped with the node — no git, no pip for fish_speech itself.
    """
    if not _FISH_SRC.is_dir():
        logger.error(
            f"fish_speech_src/ not found at {_FISH_SRC}\n"
            "The bundled fish-speech source is missing from the node folder."
        )
        return False

    fish_src_str = str(_FISH_SRC)
    if fish_src_str not in sys.path:
        sys.path.insert(0, fish_src_str)

    try:
        import fish_speech.models  # noqa: F401
        return True
    except ImportError as e:
        logger.error(f"fish_speech not importable from {_FISH_SRC}: {e}")
        return False

# After installing fish-speech we must restore the correct torch build.
# fish-speech pins torch==2.8.0 which would downgrade and break ComfyUI.
# We detect the current torch version and re-pin it with the right CUDA index.
def _restore_torch() -> None:
    """Re-install torch/torchaudio with CUDA after fish-speech may have downgraded it."""
    try:
        import torch
        version = torch.__version__
        # If it's already a CUDA build (contains +cu) we're fine
        if "+cu" in version:
            logger.info(f"torch {version} is a CUDA build — no restore needed.")
            return
        logger.warning(
            f"torch {version} is NOT a CUDA build — fish-speech downgraded it. "
            "Restoring CUDA torch..."
        )
    except ImportError:
        logger.warning("torch not found — skipping restore.")
        return

    # Detect CUDA version from nvidia-smi or fall back to cu128
    cuda_tag = "cu128"
    try:
        import subprocess as sp
        r = sp.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                   capture_output=True, text=True, timeout=5)
        # Rough mapping: driver >= 528 → cu12x
        cuda_tag = "cu128"
    except Exception:
        pass

    index_url = f"https://download.pytorch.org/whl/{cuda_tag}"
    logger.info(f"Restoring torch with: --index-url {index_url}")
    _pip_install(f"torch torchaudio --index-url {index_url}")


def _ensure_dependencies() -> bool:
    """Auto-install any missing packages. Returns True when all are available."""
    all_ok = True
    any_installed = False
    failed_specs: list[str] = []

    for import_name, pip_spec in _REQUIRED:
        try:
            __import__(import_name)
        except ImportError as e:
            logger.warning(
                f"'{import_name}' not found — auto-installing from: {pip_spec}\n"
                f"  ImportError: {e}\n"
                f"  sys.path: {sys.path}\n"
                f"  sys.modules entry: {sys.modules.get(import_name, '<not in sys.modules>')}"
            )
            if _pip_install(pip_spec):
                any_installed = True
                try:
                    __import__(import_name)
                except ImportError as e2:
                    logger.error(
                        f"Installed '{pip_spec}' but '{import_name}' still "
                        f"cannot be imported. Please restart ComfyUI.\n"
                        f"  ImportError: {e2}\n"
                        f"  sys.path: {sys.path}\n"
                        f"  sys.modules entry: {sys.modules.get(import_name, '<not in sys.modules>')}"
                    )
                    failed_specs.append(pip_spec)
                    all_ok = False
            else:
                failed_specs.append(pip_spec)
                all_ok = False

    # If any package was auto-installed, ensure torch is still a CUDA build.
    # pip may silently install a CPU torch as a transitive dependency of packages
    # like transformers or bitsandbytes — especially in embedded Python where the
    # CUDA torch was not installed via pip and has no pip metadata record.
    if any_installed:
        _restore_torch()

    if not all_ok:
        install_cmds = "\n".join(
            f"  {sys.executable} -m pip install {s}" for s in failed_specs
        )
        logger.error(
            "Auto-install failed for some packages. "
            "Install them manually then restart ComfyUI:\n"
            + install_cmds
        )
    return all_ok


# ---------------------------------------------------------------------------
# Node registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS: Dict[str, Any] = {}
NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {}

if _ensure_fish_source() and _ensure_dependencies():
    try:
        from .nodes.loader import _register_folder
        _register_folder()

        from .nodes.tts_node import FishS2TTS
        from .nodes.voice_clone_node import FishS2VoiceCloneTTS
        from .nodes.multi_speaker_node import FishS2MultiSpeakerTTS
        from .nodes.multi_speaker_split_node import FishS2MultiSpeakerSplitTTS

        NODE_CLASS_MAPPINGS = {
            "FishS2TTS": FishS2TTS,
            "FishS2VoiceCloneTTS": FishS2VoiceCloneTTS,
            "FishS2MultiSpeakerTTS": FishS2MultiSpeakerTTS,
            "FishS2MultiSpeakerSplitTTS": FishS2MultiSpeakerSplitTTS,
        }

        NODE_DISPLAY_NAME_MAPPINGS = {
            "FishS2TTS": "Fish S2 TTS",
            "FishS2VoiceCloneTTS": "Fish S2 Voice Clone TTS",
            "FishS2MultiSpeakerTTS": "Fish S2 Multi-Speaker TTS",
            "FishS2MultiSpeakerSplitTTS": "Fish S2 Multi-Speaker Split TTS",
        }

        logger.info(
            f"Registered {len(NODE_CLASS_MAPPINGS)} nodes "
            f"(v{__version__}): {', '.join(NODE_DISPLAY_NAME_MAPPINGS.values())}"
        )

    except Exception as e:
        logger.error(f"Failed to register nodes: {e}", exc_info=True)
else:
    logger.warning(
        "FishAudioS2 nodes not registered — "
        "fix dependency errors above and restart ComfyUI."
    )

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "__version__"]
