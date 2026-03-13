"""Fish Audio S2 - Text-to-Speech node."""

import logging
from typing import Tuple

from .loader import get_model_names, load_engine, numpy_audio_to_comfy
from .model_cache import (
    cancel_event,
    get_cache_key,
    get_cached_engine,
    is_offloaded,
    offload_engine_to_cpu,
    resume_engine_to_cuda,
    set_cached_engine,
    unload_engine,
)

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

LANGUAGES = [
    "auto", "en", "zh", "ja", "ko", "es", "pt", "ar", "ru", "fr", "de",
    "it", "tr", "nl", "sv", "no", "da", "fi", "pl", "hi", "vi", "th",
    "id", "ms", "uk", "bg", "hr", "cs", "sk", "sl", "ro", "hu", "et",
    "lv", "lt", "el", "he", "fa", "bn", "ta", "te", "kn", "ml", "si",
    "my", "km", "am", "ka", "az", "kk", "mn", "sw", "yo", "eu", "ca",
    "gl", "cy", "la", "sa", "ur", "ne", "tl", "jw",
]

# Shared input field definitions — used identically across all three nodes.
COMMON_GENERATION_INPUTS = {
    "max_new_tokens": ("INT", {
        "default": 0,
        "min": 0,
        "max": 4096,
        "step": 64,
        "tooltip": (
            "Maximum acoustic tokens to generate. "
            "0 = auto (no limit, model decides)."
        ),
    }),
    "chunk_length": ("INT", {
        "default": 200,
        "min": 100,
        "max": 400,
        "step": 10,
        "tooltip": (
            "Chunk length for iterative synthesis (100-400). "
            "Lower = faster first audio, slightly lower quality. "
            "Higher = better prosody across long sentences."
        ),
    }),
    "temperature": ("FLOAT", {
        "default": 0.8,
        "min": 0.1,
        "max": 1.0,
        "step": 0.05,
        "tooltip": "Sampling temperature. Lower = more deterministic output.",
    }),
    "top_p": ("FLOAT", {
        "default": 0.8,
        "min": 0.1,
        "max": 1.0,
        "step": 0.05,
        "tooltip": "Top-p nucleus sampling cutoff.",
    }),
    "repetition_penalty": ("FLOAT", {
        "default": 1.1,
        "min": 0.9,
        "max": 2.0,
        "step": 0.05,
        "tooltip": "Penalises repeated tokens. Higher = less repetition.",
    }),
    "seed": ("INT", {
        "default": 0,
        "min": 0,
        "max": 2**31 - 1,
        "tooltip": "Random seed.",
    }),
}


class FishS2TTS:
    """
    Fish Audio S2 Text-to-Speech.
    Synthesises speech from text using the S2-Pro model.
    Supports 80+ languages and inline emotion/prosody tags like [laugh], [whisper].
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
                    "default": "Hello! [excited] This is Fish Audio S2.",
                    "tooltip": (
                        "Text to synthesise. Supports inline emotion tags like "
                        "[laugh], [whisper], [pause], [excited], [sad], [angry], "
                        "[volume up], [pitch up], etc."
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
                        "BNB models (s2-pro-bnb-int8/nf4) always use sdpa regardless of this setting. "
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
                "offload_to_cpu": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "After generation, move the model to CPU instead of "
                        "keeping it in VRAM. Frees VRAM while avoiding the "
                        "full reload penalty. Slower than keep_model_loaded "
                        "but faster than a cold load. Ignored if "
                        "keep_model_loaded is OFF."
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
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "FishAudioS2"
    DESCRIPTION = (
        "Fish Audio S2-Pro TTS. Synthesises speech from text with "
        "inline emotion/prosody tags. 80+ languages."
    )

    def generate(
        self,
        model_path: str,
        text: str,
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
        offload_to_cpu: bool,
        compile_model: bool,
    ) -> Tuple[dict]:
        cancel_event.clear()
        self._check_interrupt()

        if not text.strip():
            raise ValueError("Text cannot be empty.")

        engine = self._get_engine(model_path, device, precision, attention, compile_model, keep_model_loaded, offload_to_cpu)

        from fish_speech.utils.schema import ServeTTSRequest

        pbar = ProgressBar(3) if _PBAR else None

        prompt_text = f"[{language}] {text}" if language != "auto" else text
        actual_seed = seed
        tokens = max_new_tokens if max_new_tokens > 0 else 0

        request = ServeTTSRequest(
            text=prompt_text,
            references=[],
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
            pbar.update_absolute(1, 3)

        self._check_interrupt()

        logger.info(f"TTS: {text[:80]}{'...' if len(text) > 80 else ''}")
        audio_out = None
        sample_rate = 44100

        try:
            for result in engine.inference(request):
                self._check_interrupt()
                if result.code == "error":
                    raise RuntimeError(f"Fish S2 error: {result.error}")
                if result.code == "final":
                    sample_rate, audio_out = result.audio

            if pbar:
                pbar.update_absolute(2, 3)

            if audio_out is None:
                raise RuntimeError("No audio produced.")

            output = numpy_audio_to_comfy(audio_out, sample_rate)

            if pbar:
                pbar.update_absolute(3, 3)

        finally:
            # Always run on completion, cancellation, or error.
            if not keep_model_loaded:
                unload_engine()
            elif offload_to_cpu:
                offload_engine_to_cpu()

        return (output,)

    def _get_engine(self, model_path, device, precision, attention, compile_model, keep_loaded=False, offload_to_cpu=False):
        from .loader import resolve_device
        key = get_cache_key(model_path, device, precision, attention, model_path)
        cached_engine, cached_key = get_cached_engine()
        if cached_engine is not None and cached_key == key:
            if is_offloaded():
                device_str, _ = resolve_device(device)
                logger.info(f"Resuming offloaded engine to {device_str}...")
                resume_engine_to_cuda(device_str)
            else:
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
                cancel_event.set()
                raise
