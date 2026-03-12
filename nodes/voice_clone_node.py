"""Fish Audio S2 - Voice Clone TTS node."""

import logging
from typing import Tuple

from .loader import (
    audio_bytes_from_comfy,
    get_model_names,
    load_engine,
    numpy_audio_to_comfy,
)
from .model_cache import (
    get_cache_key,
    get_cached_engine,
    set_cached_engine,
    unload_engine,
)
from .tts_node import LANGUAGES, COMMON_GENERATION_INPUTS

try:
    from comfy.utils import ProgressBar
    _PBAR = True
except ImportError:
    _PBAR = False

try:
    import comfy.model_management as mm
    _MM = True
except ImportError:
    _MM = False

logger = logging.getLogger("FishAudioS2")


class FishS2VoiceCloneTTS:
    """
    Fish Audio S2 Voice Clone TTS.
    Clones a voice from a short reference audio (10-30 s) and synthesises
    new speech in that voice. Supports inline prosody/emotion tags.
    """

    @classmethod
    def INPUT_TYPES(cls):
        model_names = get_model_names()
        return {
            "required": {
                "model_path": (model_names, {
                    "tooltip": (
                        "S2-Pro checkpoint folder name. "
                        "Place model folders in ComfyUI/models/fishaudioS2/"
                    ),
                }),
                "text": ("STRING", {
                    "multiline": True,
                    "default": "Hello! [excited] This is my cloned voice.",
                    "tooltip": (
                        "Text to synthesise in the cloned voice. "
                        "Supports inline tags: [laugh], [whisper], [pause], "
                        "[excited], [sad], [angry], [volume up], etc."
                    ),
                }),
                "reference_audio": ("AUDIO", {
                    "tooltip": (
                        "Reference audio to clone the voice from. "
                        "10-30 seconds gives the best results."
                    ),
                }),
                "language": (LANGUAGES, {
                    "default": "auto",
                    "tooltip": "Language hint. 'auto' lets the model detect it.",
                }),
                "device": (["auto", "cuda", "cpu", "mps"], {
                    "default": "auto",
                    "tooltip": "Compute device. 'auto' picks CUDA > MPS > CPU.",
                }),
                "precision": (["auto", "bfloat16", "float16", "float32"], {
                    "default": "auto",
                    "tooltip": (
                        "Model precision. 'auto' picks bfloat16 for full model, "
                        "float16 for quantized model. bfloat16 recommended for CUDA."
                    ),
                }),
                "attention": (["auto", "sdpa", "sage_attention", "flash_attention"], {
                    "default": "auto",
                    "tooltip": (
                        "Attention kernel. "
                        "'auto' uses the model default (sdpa/flash). "
                        "'sdpa' forces PyTorch SDPA. "
                        "'flash_attention' forces FlashAttention via SDPBackend. "
                        "'sage_attention' monkey-patches with SageAttention (requires sageattention). "
                        "Changing this unloads and reloads the model."
                    ),
                }),
                **COMMON_GENERATION_INPUTS,
                "keep_model_loaded": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "ON = model stays in VRAM between runs (faster). "
                        "OFF = model unloaded after each run (frees VRAM)."
                    ),
                }),
                "compile_model": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Enable torch.compile (~10x speedup after warmup). "
                        "First run is slow while compiling. "
                        "Not supported on Windows."
                    ),
                }),
            },
            "optional": {
                "reference_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "Transcript of the reference audio. "
                        "Providing this improves voice clone accuracy. "
                        "Leave blank to let the model handle it."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "FishAudioS2"
    DESCRIPTION = (
        "Fish Audio S2-Pro Voice Clone TTS. Clones any voice from a short "
        "reference audio clip and synthesises new speech in that voice."
    )

    def generate(
        self,
        model_path: str,
        text: str,
        reference_audio: dict,
        language: str,
        device: str,
        precision: str,
        attention: str,
        max_new_tokens: int,
        chunk_length: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        seed: int,
        keep_model_loaded: bool,
        compile_model: bool,
        reference_text: str = "",
    ) -> Tuple[dict]:
        self._check_interrupt()

        if not text.strip():
            raise ValueError("Text cannot be empty.")

        engine = self._get_engine(model_path, device, precision, attention, compile_model, keep_model_loaded)

        from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest

        pbar = ProgressBar(4) if _PBAR else None

        logger.info("Encoding reference audio...")
        ref_bytes = audio_bytes_from_comfy(reference_audio)

        if pbar:
            pbar.update_absolute(1, 4)

        prompt_text = f"[{language}] {text}" if language != "auto" else text
        actual_seed = seed
        tokens = max_new_tokens if max_new_tokens > 0 else 0

        request = ServeTTSRequest(
            text=prompt_text,
            references=[
                ServeReferenceAudio(
                    audio=ref_bytes,
                    text=reference_text.strip(),
                )
            ],
            reference_id=None,
            max_new_tokens=tokens,
            chunk_length=chunk_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            seed=actual_seed,
            streaming=False,
            format="wav",
        )

        if pbar:
            pbar.update_absolute(2, 4)

        self._check_interrupt()

        logger.info(f"Voice-clone TTS: {text[:80]}{'...' if len(text) > 80 else ''}")
        audio_out = None
        sample_rate = 44100

        for result in engine.inference(request):
            self._check_interrupt()
            if result.code == "error":
                raise RuntimeError(f"Fish S2 error: {result.error}")
            if result.code == "final":
                sample_rate, audio_out = result.audio

        if pbar:
            pbar.update_absolute(3, 4)

        if audio_out is None:
            raise RuntimeError("No audio produced.")

        output = numpy_audio_to_comfy(audio_out, sample_rate)

        if pbar:
            pbar.update_absolute(4, 4)

        if not keep_model_loaded:
            unload_engine()

        return (output,)

    def _get_engine(self, model_path, device, precision, attention, compile_model, keep_loaded=False):
        key = get_cache_key(model_path, device, precision, attention)
        cached_engine, cached_key = get_cached_engine()
        if cached_engine is not None and cached_key == key:
            logger.info("Reusing cached Fish S2 engine.")
            return cached_engine
        if cached_engine is not None:
            unload_engine()
        engine = load_engine(model_path, device, precision, attention, compile_model)
        set_cached_engine(engine, key, keep_loaded=keep_loaded)
        return engine

    def _check_interrupt(self):
        if _MM:
            try:
                mm.throw_exception_if_processing_interrupted()
            except Exception:
                raise
