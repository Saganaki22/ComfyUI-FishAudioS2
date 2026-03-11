"""ComfyUI custom nodes for Fish Audio S2-Pro TTS.

Provides three nodes:
  - FishS2TTS             — text → speech, 80+ languages, inline emotion tags
  - FishS2VoiceCloneTTS   — reference audio + text → cloned-voice speech
  - FishS2MultiSpeakerTTS — multi-speaker conversation synthesis in one pass

Required pip packages are auto-installed on startup.
Model weights are auto-downloaded from HuggingFace on first inference.
"""

__version__ = "0.2.1"

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
_GPTQ_SRC = _HERE / "auto_gptq_src"

# Add fish_speech_src to path
if _FISH_SRC.is_dir():
    _fish_src_str = str(_FISH_SRC)
    if _fish_src_str not in sys.path:
        sys.path.insert(0, _fish_src_str)

# Add auto_gptq_src to path (for GPTQ quantized model support)
if _GPTQ_SRC.is_dir():
    _gptq_src_str = str(_GPTQ_SRC)
    if _gptq_src_str not in sys.path:
        sys.path.insert(0, _gptq_src_str)

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
    Return the pip command that installs into the same environment as the
    currently-running Python — regardless of install type.

    Portable embedded:  python_embeded/python.exe -m pip
    venv / conda:       <venv>/bin/python -m pip
    System Python:      python -m pip

    Using [sys.executable, "-m", "pip"] is the only reliable method because:
    - It always targets the active interpreter, not any pip.exe on PATH
    - It works even when pip.exe doesn't exist but pip is installed as a module
    - It works inside embedded Python where Scripts/ may not be on PATH
    """
    return [sys.executable, "-m", "pip"]


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
            timeout=300,
        )
        if result.returncode == 0:
            logger.info(f"Successfully installed: {spec}")
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
    ("google.protobuf", "protobuf>=4.21.0"),
    ("safetensors",     "safetensors>=0.4.0"),
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
    fish_speech_was_missing = False

    for import_name, pip_spec in _REQUIRED:
        try:
            __import__(import_name)
        except ImportError:
            logger.warning(
                f"'{import_name}' not found — auto-installing from: {pip_spec}"
            )
            if import_name == "fish_speech":
                fish_speech_was_missing = True
            if _pip_install(pip_spec):
                try:
                    __import__(import_name)
                except ImportError:
                    logger.error(
                        f"Installed '{pip_spec}' but '{import_name}' still "
                        "cannot be imported. Please restart ComfyUI."
                    )
                    all_ok = False
            else:
                all_ok = False

    # If we just installed fish-speech, restore torch in case it was downgraded
    if fish_speech_was_missing:
        _restore_torch()

    if not all_ok:
        logger.error(
            "Auto-install failed for some packages. "
            "Install them manually then restart ComfyUI:\n"
            f"  {sys.executable} -m pip install soundfile\n"
            f"  {sys.executable} -m pip install "
            "git+https://github.com/fishaudio/fish-speech"
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

        NODE_CLASS_MAPPINGS = {
            "FishS2TTS": FishS2TTS,
            "FishS2VoiceCloneTTS": FishS2VoiceCloneTTS,
            "FishS2MultiSpeakerTTS": FishS2MultiSpeakerTTS,
        }

        NODE_DISPLAY_NAME_MAPPINGS = {
            "FishS2TTS": "Fish S2 TTS",
            "FishS2VoiceCloneTTS": "Fish S2 Voice Clone TTS",
            "FishS2MultiSpeakerTTS": "Fish S2 Multi-Speaker TTS",
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
