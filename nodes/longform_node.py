"""Fish Audio S2 - Long-Form TTS node with native chunking and sliding window context."""

import logging
import numpy as np
import torch
from typing import Tuple

from .loader import (
    audio_bytes_from_comfy,
    get_model_names,
    load_engine,
    numpy_audio_to_comfy,
)
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
from .tts_node import LANGUAGES, COMMON_GENERATION_INPUTS
from .utils import split_text_into_chunks

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


class FishS2LongFormTTS:
    """
    Fish Audio S2 Long-Form TTS with native chunking.
    Uses <|speaker:0|> tags for native batching with sliding window context
    to maintain voice consistency across long texts.
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
                "default": "This is a long text that will be split into multiple chunks for processing. "
                "Each chunk maintains context from previous chunks for consistent voice.",
                "tooltip": (
                    "Long text to synthesise. Automatically split into chunks "
                    "at sentence boundaries for processing."
                ),
            }),
                "language": (LANGUAGES, {
                "default": "auto",
                "tooltip": "Language hint. 'auto' detects automatically.",
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
                    "Attention kernel. 'auto' uses model default. "
                    "BNB models always use sdpa regardless of this setting."
                ),
            }),
            **COMMON_GENERATION_INPUTS,
            "max_context_batches": ("INT", {
                "default": 3,
                "min": -1,
                "max": 20,
                "tooltip": (
                    "Previous batches to keep as context for voice consistency. "
                    "0 = no context (each chunk independent, safest for low VRAM). "
                    "3 = recommended for 12GB+ VRAM. "
                    "-1 = unlimited (may OOM on very long texts). "
                    "Auto-set to 0 when BNB/low VRAM detected."
                ),
            }),
            "max_words_per_chunk": ("INT", {
                "default": 150,
                "min": 50,
                "max": 400,
                "tooltip": (
                    "Maximum words per chunk when splitting text. "
                    "Lower = more chunks but better granularity. "
                    "Higher = fewer chunks but may exceed context limits."
                ),
            }),
            "low_vram_mode": ("BOOLEAN", {
                "default": False,
                "tooltip": (
                    "Enable for low-VRAM GPUs (8GB). Does two things: "
                    "(1) Offloads decoder/LLaMA during inference to prevent OOM. "
                    "(2) Makes each chunk independent (no context accumulation). "
                    "Slower but safer for long texts on limited VRAM."
                ),
            }),
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
                    "full reload penalty."
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
            "reference_audio": ("AUDIO", {
                "tooltip": (
                    "Reference audio to clone the voice from (10-30 seconds). "
                    "Leave disconnected for random/default voice."
                ),
            }),
            "reference_text": ("STRING", {
                "multiline": True,
                "default": "",
                "tooltip": (
                    "Transcript of the reference audio. "
                    "Improves voice clone accuracy. Leave blank if no reference audio."
                ),
            }),
        },
    }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "FishAudioS2"
    DESCRIPTION = (
        "Fish Audio S2 Long-Form TTS. Synthesises long texts with native chunking "
        "and sliding window context for consistent voice across chunks."
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
        max_context_batches: int,
        max_words_per_chunk: int,
        low_vram_mode: bool,
        keep_model_loaded: bool,
        offload_to_cpu: bool,
        compile_model: bool,
        reference_audio: dict = None,
        reference_text: str = "",
    ) -> Tuple[dict]:
        cancel_event.clear()
        self._check_interrupt()

        if not text.strip():
            raise ValueError("Text cannot be empty.")

        engine = self._get_engine(
            model_path, device, precision, attention, compile_model, keep_model_loaded, offload_to_cpu
        )

        from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest

        has_reference = reference_audio is not None
        references = []
        if has_reference:
            logger.info("Encoding reference audio...")
            ref_bytes = audio_bytes_from_comfy(reference_audio)
            references = [
                ServeReferenceAudio(
                    audio=ref_bytes,
                    text=reference_text.strip(),
                )
            ]

        text_chunks = split_text_into_chunks(text, max_words_per_chunk=max_words_per_chunk)
        num_chunks = len(text_chunks)

        # Auto-detect low VRAM and BNB models for context management
        is_bnb_model = "bnb" in model_path.lower()
        vram_gb = 0
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        is_low_vram = vram_gb > 0 and vram_gb < 10

        # For BNB, low VRAM, or low_vram_mode enabled: force no context accumulation to prevent OOM
        # This makes each batch independent (fresh start) - no KV cache growth across batches
        effective_max_context = max_context_batches
        if is_bnb_model or is_low_vram or low_vram_mode:
            if max_context_batches == -1 or max_context_batches > 0:
                reason = "BNB" if is_bnb_model else ("Low VRAM" if is_low_vram else "low_vram_mode")
                logger.info(
                    f"{reason}{' (' + str(round(vram_gb, 1)) + 'GB)' if vram_gb > 0 else ''} — "
                    f"limiting context batches: {max_context_batches} → 0 (each batch independent)"
                )
                effective_max_context = 0

        logger.info(f"Long-form TTS: {num_chunks} chunks (max_context_batches={effective_max_context})")

        pbar = ProgressBar(num_chunks) if _PBAR else None

        audio_segments = []
        sample_rate = 44100
        tokens = max_new_tokens

        combined_text = "\n".join(text_chunks)
        if language != "auto":
            combined_text = f"[{language}] {combined_text}"

        request = ServeTTSRequest(
            text=combined_text,
            references=references,
            reference_id=None,
            max_new_tokens=tokens,
            chunk_length=chunk_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            seed=seed if seed else None,
            streaming=False,
            format="wav",
            max_context_batches=effective_max_context,
            low_vram_mode=low_vram_mode,
        )

        self._check_interrupt()

        logger.info(f"Long-form TTS: {text[:80]}{'...' if len(text) > 80 else ''}")
        audio_out = None

        try:
            for result in engine.inference(request):
                self._check_interrupt()
                if result.code == "error":
                    raise RuntimeError(f"Fish S2 error: {result.error}")
                if result.code == "segment":
                    if pbar:
                        pbar.update_absolute(1, num_chunks)
                elif result.code == "final":
                    sample_rate, audio_out = result.audio

            if audio_out is None:
                raise RuntimeError("No audio produced.")

            output = numpy_audio_to_comfy(audio_out, sample_rate)

        finally:
            if not keep_model_loaded:
                unload_engine()
            elif offload_to_cpu:
                offload_engine_to_cpu()

        return (output,)

    def _get_engine(self, model_path, device, precision, attention, compile_model, keep_loaded=False, offload_to_cpu=False):
        from .loader import resolve_device, _strip_auto_download_suffix
        model_name = _strip_auto_download_suffix(model_path)
        key = get_cache_key(model_path, device, precision, attention, model_name)
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
