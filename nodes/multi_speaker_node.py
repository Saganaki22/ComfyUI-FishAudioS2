"""Fish Audio S2 - Multi-Speaker TTS node with dynamic speaker inputs.

Uses the ComfyUI v3 IO API (IO.ComfyNode + DynamicCombo) so that the
speaker_N_audio / speaker_N_ref_text inputs appear and disappear as the
user changes num_speakers — only the inputs for the selected count are
shown, not all 10 at once.
"""

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

try:
    from comfy_api.latest import IO
    _V3 = True
except ImportError:
    _V3 = False

logger = logging.getLogger("FishAudioS2")

MAX_SPEAKERS = 10


# ---------------------------------------------------------------------------
# Helper — build the per-option input list for a given speaker count
# ---------------------------------------------------------------------------

def _speaker_inputs(count: int) -> list:
    """Return IO input descriptors for `count` speakers (1-indexed for UI)."""
    inputs = []
    for i in range(1, count + 1):
        inputs.append(
            IO.Audio.Input(
                f"speaker_{i}_audio",
                optional=True,
                tooltip=(
                    f"Reference audio for speaker {i}. "
                    f"Use [speaker_{i}]: in your text for this voice."
                ),
            )
        )
        inputs.append(
            IO.String.Input(
                f"speaker_{i}_ref_text",
                multiline=False,
                default="",
                optional=True,
                tooltip=(
                    f"Optional transcript of speaker {i}'s reference audio. "
                    "Providing it improves clone accuracy."
                ),
            )
        )
    return inputs


def _convert_speaker_tags(text: str) -> str:
    """Convert user-friendly [speaker_N]: tags to model's <|speaker:N-1|> format."""
    import re
    
    def replace_tag(m):
        n = int(m.group(1))
        colon = m.group(2) or ""
        return f"<|speaker:{n - 1}|>{colon}"
    
    return re.sub(r'\[speaker_(\d+)\](:)?', replace_tag, text)


# ---------------------------------------------------------------------------
# V3 node (DynamicCombo — inputs update when num_speakers changes)
# ---------------------------------------------------------------------------

if _V3:
    class FishS2MultiSpeakerTTS(IO.ComfyNode):
        """
        Fish Audio S2 Multi-Speaker TTS.
        Synthesises a conversation with multiple cloned voices in one pass.
        Change num_speakers to show/hide speaker reference audio inputs.
        Use <|speaker:0|>, <|speaker:1|>, ... tokens in the text.
        """

        @classmethod
        def define_schema(cls) -> IO.Schema:
            model_names = get_model_names()

            # One DynamicCombo option per speaker count (2..MAX_SPEAKERS)
            speaker_options = [
                IO.DynamicCombo.Option(
                    key=str(n),
                    inputs=_speaker_inputs(n),
                )
                for n in range(2, MAX_SPEAKERS + 1)
            ]

            return IO.Schema(
                node_id="FishS2MultiSpeakerTTS",
                display_name="Fish S2 Multi-Speaker TTS",
                category="FishAudioS2",
                description=(
                    "Fish Audio S2-Pro Multi-Speaker TTS. Synthesises a "
                    "conversation between multiple cloned voices in one "
                    "generation pass. Connect reference audio clips and use "
                    "<|speaker:N|> tokens in text."
                ),
                inputs=[
                    IO.Combo.Input(
                        "model_path",
                        options=model_names,
                        tooltip=(
                            "S2-Pro checkpoint folder name. "
                            "Place model folders in ComfyUI/models/fishaudioS2/"
                        ),
                    ),
                    IO.String.Input(
                        "text",
                        multiline=True,
                        default=(
                            "[speaker_1]: Hello, I'm speaker one.\n"
                            "[speaker_2]: And I'm speaker two!"
                        ),
                        tooltip=(
                            "Multi-speaker text. Use [speaker_1]:, "
                            "[speaker_2]:, ... to assign lines to each "
                            "connected speaker. Supports inline tags: "
                            "[laugh], [whisper], etc."
                        ),
                    ),
                    IO.Combo.Input(
                        "language",
                        options=LANGUAGES,
                        tooltip="Language hint. 'auto' lets the model detect it.",
                    ),
                    IO.Combo.Input(
                        "device",
                        options=["auto", "cuda", "cpu", "mps"],
                        tooltip="Compute device. 'auto' picks CUDA > MPS > CPU.",
                    ),
                    IO.Combo.Input(
                        "precision",
                        options=["auto", "bfloat16", "float16", "float32"],
                        tooltip=(
                            "Model precision. 'auto' picks bfloat16 for full model, "
                            "float16 for quantized model. bfloat16 recommended for CUDA."
                        ),
                    ),
                    IO.Combo.Input(
                        "attention",
                        options=["auto", "sdpa", "sage_attention", "flash_attention"],
                        tooltip=(
                            "Attention kernel. "
                            "'auto' uses model default. "
                            "'sdpa' forces PyTorch SDPA. "
                            "'flash_attention' forces FlashAttention. "
                            "'sage_attention' requires sageattention package. "
                            "Changing this reloads the model."
                        ),
                    ),
                    IO.Int.Input(
                        "max_new_tokens",
                        default=0, min=0, max=4096, step=64,
                        tooltip="Max acoustic tokens. 0 = auto.",
                    ),
                    IO.Int.Input(
                        "chunk_length",
                        default=200, min=100, max=400, step=10,
                        tooltip="Chunk length for iterative synthesis (100-400).",
                    ),
                    IO.Float.Input(
                        "temperature",
                        default=0.8, min=0.1, max=1.0, step=0.05,
                        tooltip="Sampling temperature.",
                    ),
                    IO.Float.Input(
                        "top_p",
                        default=0.8, min=0.1, max=1.0, step=0.05,
                        tooltip="Top-p nucleus sampling cutoff.",
                    ),
                    IO.Float.Input(
                        "repetition_penalty",
                        default=1.1, min=0.9, max=2.0, step=0.05,
                        tooltip="Repetition penalty. Higher = less repetition.",
                    ),
                    IO.Int.Input(
                        "seed",
                        default=0, min=0, max=2**31 - 1,
                        tooltip="Random seed.",
                    ),
                    IO.Boolean.Input(
                        "keep_model_loaded",
                        default=True,
                        tooltip=(
                            "ON = model stays in VRAM between runs. "
                            "OFF = unloaded after each run."
                        ),
                    ),
                    IO.Boolean.Input(
                        "compile_model",
                        default=False,
                        tooltip=(
                            "torch.compile for ~10x speedup after warmup. "
                            "Not supported on Windows."
                        ),
                    ),
                    IO.Float.Input(
                        "pause_after_speaker",
                        default=0.4, min=0.0, max=2.0, step=0.1,
                        tooltip="Seconds of silence to add after each speaker turn.",
                    ),
                    IO.DynamicCombo.Input(
                        "num_speakers",
                        options=speaker_options,
                        display_name="Number of Speakers",
                        tooltip=(
                            f"How many speakers (2-{MAX_SPEAKERS}). "
                            "Changing this shows/hides speaker audio inputs."
                        ),
                    ),
                ],
                outputs=[
                    IO.Audio.Output(display_name="audio"),
                ],
            )

        @classmethod
        def execute(
            cls,
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
            compile_model: bool,
            pause_after_speaker: float,
            num_speakers: dict,
        ) -> IO.NodeOutput:
            _check_interrupt()

            if not text.strip():
                raise ValueError("Text cannot be empty.")

            engine = _get_engine(model_path, device, precision, attention, compile_model, keep_model_loaded)

            from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest

            # num_speakers is a dict from DynamicCombo:
            # {"num_speakers": "3", "speaker_1_audio": ..., "speaker_1_ref_text": ..., ...}
            n = int(num_speakers["num_speakers"])

            total_steps = 2 + n + 1
            pbar = ProgressBar(total_steps) if _PBAR else None
            step = 0

            references = []
            missing = []
            for i in range(1, n + 1):
                speaker_audio = num_speakers.get(f"speaker_{i}_audio")
                speaker_ref_text = num_speakers.get(f"speaker_{i}_ref_text") or ""

                if speaker_audio is None:
                    missing.append(i)
                    references.append(None)
                else:
                    logger.info(f"Encoding reference audio for speaker {i}...")
                    ref_bytes = audio_bytes_from_comfy(speaker_audio)
                    logger.debug(f"Speaker {i} audio bytes: {len(ref_bytes)}")
                    references.append(
                        ServeReferenceAudio(
                            audio=ref_bytes,
                            text=speaker_ref_text.strip(),
                        )
                    )

                step += 1
                if pbar:
                    pbar.update_absolute(step, total_steps)

            if missing:
                missing_str = ", ".join(f"speaker_{i}_audio" for i in missing)
                raise ValueError(
                    f"Reference audio required for all speakers. "
                    f"Missing: {missing_str}. "
                    f"Please connect reference audio clips to each speaker input."
                )

            _check_interrupt()

            prompt_text = _convert_speaker_tags(text)
            if language != "auto":
                prompt_text = f"[{language}] {prompt_text}"
            actual_seed = seed
            tokens = max_new_tokens if max_new_tokens > 0 else 0

            request = ServeTTSRequest(
                text=prompt_text,
                references=references,
                reference_id=None,
                max_new_tokens=tokens,
                chunk_length=chunk_length,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                seed=actual_seed,
                streaming=True,
                format="wav",
            )

            step += 1
            if pbar:
                pbar.update_absolute(step, total_steps)

            logger.info(
                f"Multi-speaker TTS ({n} speakers): "
                f"{text[:80]}{'...' if len(text) > 80 else ''}"
            )
            segments = []
            sample_rate = 44100

            for result in engine.inference(request):
                if result.code == "error":
                    raise RuntimeError(f"Fish S2 error: {result.error}")
                if result.code == "segment":
                    sr, seg = result.audio
                    segments.append((sr, seg))
                if result.code == "final":
                    sample_rate, audio_out = result.audio

            if len(segments) == 0 and audio_out is None:
                raise RuntimeError("No audio produced.")

            if len(segments) > 0 and pause_after_speaker > 0:
                import numpy as np
                silence_samples = int(pause_after_speaker * sample_rate)
                silence = np.zeros(silence_samples, dtype=np.float32)
                audio_segments = [s[1] for s in segments]
                audio_out = audio_segments[0]
                for seg in audio_segments[1:]:
                    audio_out = np.concatenate([audio_out, silence, seg], axis=0)
            elif audio_out is None:
                audio_out = np.concatenate([s[1] for s in segments], axis=0)

            output = numpy_audio_to_comfy(audio_out, sample_rate)

            step += 1
            if pbar:
                pbar.update_absolute(step, total_steps)

            if not keep_model_loaded:
                unload_engine()

            return IO.NodeOutput(output)

# ---------------------------------------------------------------------------
# V2 fallback (old INPUT_TYPES API) — used if ComfyUI < 0.8.1
# Keeps all 10 speaker slots always visible (original behaviour).
# ---------------------------------------------------------------------------

else:
    class FishS2MultiSpeakerTTS:  # type: ignore[no-redef]
        """
        Fish Audio S2 Multi-Speaker TTS (legacy fallback — upgrade ComfyUI
        to 0.8.1+ for dynamic speaker inputs).
        """

        @classmethod
        def INPUT_TYPES(cls):
            model_names = get_model_names()
            optional_inputs = {}
            for i in range(1, MAX_SPEAKERS + 1):
                optional_inputs[f"speaker_{i}_audio"] = ("AUDIO", {
                    "tooltip": (
                        f"Reference audio for speaker {i}. "
                        f"Use [speaker_{i}]: in text."
                    ),
                })
                optional_inputs[f"speaker_{i}_ref_text"] = ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": f"Optional transcript of speaker {i}'s reference audio.",
                })

            return {
                "required": {
                    "model_path": (model_names, {}),
                    "text": ("STRING", {
                        "multiline": True,
                        "default": (
                            "[speaker_1]: Hello, I'm speaker one.\n"
                            "[speaker_2]: And I'm speaker two!"
                        ),
                    }),
                    "num_speakers": ("INT", {
                        "default": 2, "min": 2, "max": MAX_SPEAKERS, "step": 1,
                        "tooltip": f"Number of active speakers (2-{MAX_SPEAKERS}).",
                    }),
                    "language": (LANGUAGES, {"default": "auto"}),
                    "device": (["auto", "cuda", "cpu", "mps"], {"default": "auto"}),
                    "precision": (["auto", "bfloat16", "float16", "float32"], {"default": "auto"}),
                    "attention": (["auto", "sdpa", "sage_attention", "flash_attention"], {"default": "auto"}),
                    **COMMON_GENERATION_INPUTS,
                    "pause_after_speaker": ("FLOAT", {
                        "default": 0.4, "min": 0.0, "max": 5.0, "step": 0.1,
                        "tooltip": "Seconds of silence to add after each speaker's turn.",
                    }),
                    "keep_model_loaded": ("BOOLEAN", {"default": True}),
                    "compile_model": ("BOOLEAN", {"default": False}),
                },
                "optional": optional_inputs,
            }

        RETURN_TYPES = ("AUDIO",)
        RETURN_NAMES = ("audio",)
        FUNCTION = "generate"
        CATEGORY = "FishAudioS2"
        DESCRIPTION = "Fish Audio S2-Pro Multi-Speaker TTS (legacy mode — upgrade ComfyUI for dynamic inputs)."

        def generate(
            self,
            model_path, text, num_speakers, language, device, precision, attention,
            max_new_tokens, chunk_length, temperature, top_p, repetition_penalty,
            seed, pause_after_speaker, keep_model_loaded, compile_model, **kwargs,
        ):
            _check_interrupt()
            if not text.strip():
                raise ValueError("Text cannot be empty.")

            engine = _get_engine(model_path, device, precision, attention, compile_model, keep_model_loaded)

            from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest

            total_steps = 2 + num_speakers + 1
            pbar = ProgressBar(total_steps) if _PBAR else None
            step = 0

            references = []
            missing = []
            for i in range(1, num_speakers + 1):
                speaker_audio = kwargs.get(f"speaker_{i}_audio")
                speaker_ref_text = kwargs.get(f"speaker_{i}_ref_text") or ""
                if speaker_audio is None:
                    missing.append(i)
                    references.append(None)
                else:
                    ref_bytes = audio_bytes_from_comfy(speaker_audio)
                    logger.debug(f"Speaker {i} audio bytes: {len(ref_bytes)}")
                    references.append(
                        ServeReferenceAudio(audio=ref_bytes, text=speaker_ref_text.strip())
                    )
                step += 1
                if pbar:
                    pbar.update_absolute(step, total_steps)

            if missing:
                missing_str = ", ".join(f"speaker_{i}_audio" for i in missing)
                raise ValueError(
                    f"Reference audio required for all speakers. "
                    f"Missing: {missing_str}. "
                    f"Please connect reference audio clips to each speaker input."
                )

            _check_interrupt()
            prompt_text = _convert_speaker_tags(text)
            if language != "auto":
                prompt_text = f"[{language}] {prompt_text}"
            actual_seed = seed
            tokens = max_new_tokens if max_new_tokens > 0 else 0

            request = ServeTTSRequest(
                text=prompt_text, references=references, reference_id=None,
                max_new_tokens=tokens, chunk_length=chunk_length, top_p=top_p,
                repetition_penalty=repetition_penalty, temperature=temperature,
                seed=actual_seed, streaming=True, format="wav",
            )

            step += 1
            if pbar:
                pbar.update_absolute(step, total_steps)

            segments = []
            sample_rate = 44100
            audio_out = None
            for result in engine.inference(request):
                if result.code == "error":
                    raise RuntimeError(f"Fish S2 error: {result.error}")
                if result.code == "segment":
                    sr, seg = result.audio
                    segments.append((sr, seg))
                if result.code == "final":
                    sample_rate, audio_out = result.audio

            if len(segments) == 0 and audio_out is None:
                raise RuntimeError("No audio produced.")

            if len(segments) > 0 and pause_after_speaker > 0:
                import numpy as np
                silence_samples = int(pause_after_speaker * sample_rate)
                silence = np.zeros(silence_samples, dtype=np.float32)
                audio_segments = [s[1] for s in segments]
                audio_out = audio_segments[0]
                for seg in audio_segments[1:]:
                    audio_out = np.concatenate([audio_out, silence, seg], axis=0)
            elif audio_out is None:
                audio_out = np.concatenate([s[1] for s in segments], axis=0)

            output = numpy_audio_to_comfy(audio_out, sample_rate)
            step += 1
            if pbar:
                pbar.update_absolute(step, total_steps)

            if not keep_model_loaded:
                unload_engine()

            return (output,)


# ---------------------------------------------------------------------------
# Shared helpers (used by both the v3 class method and v2 instance method)
# ---------------------------------------------------------------------------

def _get_engine(model_path, device, precision, attention, compile_model, keep_loaded=False):
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


def _check_interrupt():
    if _MM:
        try:
            mm.throw_exception_if_processing_interrupted()
        except Exception:
            raise
