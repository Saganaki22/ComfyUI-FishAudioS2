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


def _parse_dialogue_lines(text: str):
    """
    Parse multi-speaker text into a list of (speaker_idx_0based, line_text) tuples.

    Recognises both the user-friendly form:
        [speaker_1]: Hello world
    and the model-internal form:
        <|speaker:0|>: Hello world

    Lines that do not start with a speaker tag are silently dropped.
    Lines within a single speaker turn that span multiple physical lines are
    joined back together.

    Returns: list of (int, str) — (0-based speaker index, text for that turn)
    """
    import re

    # Match [speaker_N]: or <|speaker:N|>: at the start of a logical line
    tag_re = re.compile(
        r'^\s*(?:\[speaker_(\d+)\]|<\|speaker:(\d+)\|>):?\s*(.*)$'
    )

    lines = text.splitlines()
    turns = []          # [(speaker_0based, text), ...]
    current_speaker = None
    current_parts = []

    for raw in lines:
        m = tag_re.match(raw)
        if m:
            # Flush previous turn
            if current_speaker is not None and current_parts:
                turns.append((current_speaker, " ".join(current_parts).strip()))
            # Start new turn
            # group(1) = 1-based from [speaker_N], group(2) = 0-based from <|speaker:N|>
            if m.group(1) is not None:
                current_speaker = int(m.group(1)) - 1   # convert to 0-based
            else:
                current_speaker = int(m.group(2))       # already 0-based
            current_parts = [m.group(3)] if m.group(3).strip() else []
        else:
            stripped = raw.strip()
            if stripped and current_speaker is not None:
                current_parts.append(stripped)

    # Flush last turn
    if current_speaker is not None and current_parts:
        turns.append((current_speaker, " ".join(current_parts).strip()))

    return turns


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
                            "BNB models (s2-pro-bnb-int8/nf4) always use sdpa regardless of this setting. "
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
                        "offload_to_cpu",
                        default=False,
                        tooltip=(
                            "After generation, move the model to CPU instead of "
                            "keeping it in VRAM. Frees VRAM while avoiding the "
                            "full reload penalty. Ignored if keep_model_loaded is OFF."
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
            offload_to_cpu: bool,
            compile_model: bool,
            pause_after_speaker: float,
            num_speakers: dict,
        ) -> IO.NodeOutput:
            import numpy as np

            cancel_event.clear()
            _check_interrupt()

            if not text.strip():
                raise ValueError("Text cannot be empty.")

            engine = _get_engine(model_path, device, precision, attention, compile_model, keep_model_loaded, offload_to_cpu)

            from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest

            # num_speakers is a dict from DynamicCombo:
            # {"num_speakers": "3", "speaker_1_audio": ..., "speaker_1_ref_text": ..., ...}
            n = int(num_speakers["num_speakers"])

            # Build per-speaker reference objects (index 0-based)
            references = {}   # {0-based idx: ServeReferenceAudio}
            missing = []
            for i in range(1, n + 1):
                speaker_audio = num_speakers.get(f"speaker_{i}_audio")
                speaker_ref_text = num_speakers.get(f"speaker_{i}_ref_text") or ""

                if speaker_audio is None:
                    missing.append(i)
                else:
                    logger.info(f"Encoding reference audio for speaker {i}...")
                    ref_bytes = audio_bytes_from_comfy(speaker_audio)
                    logger.debug(f"Speaker {i} audio bytes: {len(ref_bytes)}")
                    references[i - 1] = ServeReferenceAudio(
                        audio=ref_bytes,
                        text=speaker_ref_text.strip(),
                    )

            if missing:
                missing_str = ", ".join(f"speaker_{i}_audio" for i in missing)
                raise ValueError(
                    f"Reference audio required for all speakers. "
                    f"Missing: {missing_str}. "
                    f"Please connect reference audio clips to each speaker input."
                )

            _check_interrupt()

            # Parse dialogue into individual (speaker_0based, line_text) turns
            dialogue_lines = _parse_dialogue_lines(text)
            if not dialogue_lines:
                raise ValueError(
                    "No speaker lines found. Use [speaker_1]: text format."
                )

            logger.info(
                f"Multi-speaker TTS ({n} speakers): {len(dialogue_lines)} lines — "
                f"generating each line independently then concatenating."
            )

            tokens = max_new_tokens if max_new_tokens > 0 else 0
            sample_rate = 44100
            audio_turns = []   # one numpy array per dialogue line

            total_steps = len(dialogue_lines)
            pbar = ProgressBar(total_steps) if _PBAR else None

            try:
                for line_idx, (speaker_idx, line_text) in enumerate(dialogue_lines):
                    _check_interrupt()

                    if speaker_idx not in references:
                        raise ValueError(
                            f"Line {line_idx + 1} uses speaker index {speaker_idx + 1} "
                            f"but no reference audio was provided for that speaker."
                        )

                    lang_prefix = f"[{language}] " if language != "auto" else ""
                    request_text = f"{lang_prefix}{line_text}"

                    logger.info(
                        f"  Line {line_idx + 1}/{len(dialogue_lines)} "
                        f"[speaker_{speaker_idx + 1}]: {line_text[:60]}"
                        f"{'...' if len(line_text) > 60 else ''}"
                    )

                    request = ServeTTSRequest(
                        text=request_text,
                        references=[references[speaker_idx]],
                        reference_id=None,
                        max_new_tokens=tokens,
                        chunk_length=chunk_length,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        temperature=temperature,
                        seed=seed + line_idx,   # vary seed per line for naturalness
                        streaming=False,
                        format="wav",
                    )

                    line_audio = None
                    for result in engine.inference(request):
                        if result.code == "error":
                            raise RuntimeError(f"Fish S2 error on line {line_idx + 1}: {result.error}")
                        if result.code == "final":
                            sr, line_audio = result.audio
                            sample_rate = sr

                    if line_audio is None:
                        raise RuntimeError(f"No audio produced for line {line_idx + 1}.")

                    audio_turns.append(line_audio)

                    if pbar:
                        pbar.update_absolute(line_idx + 1, total_steps)

                # Concatenate all turns with optional silence between them
                if pause_after_speaker > 0:
                    silence_samples = int(pause_after_speaker * sample_rate)
                    silence = np.zeros(silence_samples, dtype=np.float32)
                    audio_out = audio_turns[0]
                    for turn in audio_turns[1:]:
                        audio_out = np.concatenate([audio_out, silence, turn], axis=0)
                else:
                    audio_out = np.concatenate(audio_turns, axis=0)

                output = numpy_audio_to_comfy(audio_out, sample_rate)

            finally:
                # Always run on completion, cancellation, or error.
                if not keep_model_loaded:
                    unload_engine()
                elif offload_to_cpu:
                    offload_engine_to_cpu()

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
                    "attention": (["auto", "sdpa", "sage_attention", "flash_attention"], {
                        "default": "auto",
                        "tooltip": "BNB models (s2-pro-bnb-int8/nf4) always use sdpa regardless of this setting.",
                    }),
                    **COMMON_GENERATION_INPUTS,
                    "pause_after_speaker": ("FLOAT", {
                        "default": 0.4, "min": 0.0, "max": 5.0, "step": 0.1,
                        "tooltip": "Seconds of silence to add after each speaker's turn.",
                    }),
                    "keep_model_loaded": ("BOOLEAN", {"default": True}),
                    "offload_to_cpu": ("BOOLEAN", {
                        "default": False,
                        "tooltip": (
                            "After generation, move the model to CPU instead of "
                            "keeping it in VRAM. Frees VRAM while avoiding the "
                            "full reload penalty. Ignored if keep_model_loaded is OFF."
                        ),
                    }),
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
            seed, pause_after_speaker, keep_model_loaded, offload_to_cpu, compile_model, **kwargs,
        ):
            import numpy as np

            cancel_event.clear()
            _check_interrupt()
            if not text.strip():
                raise ValueError("Text cannot be empty.")

            engine = _get_engine(model_path, device, precision, attention, compile_model, keep_model_loaded, offload_to_cpu)

            from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest

            # Build per-speaker reference map (0-based index)
            references = {}
            missing = []
            for i in range(1, num_speakers + 1):
                speaker_audio = kwargs.get(f"speaker_{i}_audio")
                speaker_ref_text = kwargs.get(f"speaker_{i}_ref_text") or ""
                if speaker_audio is None:
                    missing.append(i)
                else:
                    logger.info(f"Encoding reference audio for speaker {i}...")
                    ref_bytes = audio_bytes_from_comfy(speaker_audio)
                    logger.debug(f"Speaker {i} audio bytes: {len(ref_bytes)}")
                    references[i - 1] = ServeReferenceAudio(
                        audio=ref_bytes, text=speaker_ref_text.strip()
                    )

            if missing:
                missing_str = ", ".join(f"speaker_{i}_audio" for i in missing)
                raise ValueError(
                    f"Reference audio required for all speakers. "
                    f"Missing: {missing_str}. "
                    f"Please connect reference audio clips to each speaker input."
                )

            _check_interrupt()

            # Parse dialogue into individual (speaker_0based, line_text) turns
            dialogue_lines = _parse_dialogue_lines(text)
            if not dialogue_lines:
                raise ValueError(
                    "No speaker lines found. Use [speaker_1]: text format."
                )

            logger.info(
                f"Multi-speaker TTS ({num_speakers} speakers): "
                f"{len(dialogue_lines)} lines — generating each line independently."
            )

            tokens = max_new_tokens if max_new_tokens > 0 else 0
            sample_rate = 44100
            audio_turns = []

            total_steps = len(dialogue_lines)
            pbar = ProgressBar(total_steps) if _PBAR else None

            try:
                for line_idx, (speaker_idx, line_text) in enumerate(dialogue_lines):
                    _check_interrupt()

                    if speaker_idx not in references:
                        raise ValueError(
                            f"Line {line_idx + 1} uses speaker index {speaker_idx + 1} "
                            f"but no reference audio was provided for that speaker."
                        )

                    lang_prefix = f"[{language}] " if language != "auto" else ""
                    request_text = f"{lang_prefix}{line_text}"

                    logger.info(
                        f"  Line {line_idx + 1}/{len(dialogue_lines)} "
                        f"[speaker_{speaker_idx + 1}]: {line_text[:60]}"
                        f"{'...' if len(line_text) > 60 else ''}"
                    )

                    request = ServeTTSRequest(
                        text=request_text,
                        references=[references[speaker_idx]],
                        reference_id=None,
                        max_new_tokens=tokens,
                        chunk_length=chunk_length,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        temperature=temperature,
                        seed=seed + line_idx,
                        streaming=False,
                        format="wav",
                    )

                    line_audio = None
                    for result in engine.inference(request):
                        if result.code == "error":
                            raise RuntimeError(f"Fish S2 error on line {line_idx + 1}: {result.error}")
                        if result.code == "final":
                            sr, line_audio = result.audio
                            sample_rate = sr

                    if line_audio is None:
                        raise RuntimeError(f"No audio produced for line {line_idx + 1}.")

                    audio_turns.append(line_audio)

                    if pbar:
                        pbar.update_absolute(line_idx + 1, total_steps)

                # Concatenate all turns with optional silence between them
                if pause_after_speaker > 0:
                    silence_samples = int(pause_after_speaker * sample_rate)
                    silence = np.zeros(silence_samples, dtype=np.float32)
                    audio_out = audio_turns[0]
                    for turn in audio_turns[1:]:
                        audio_out = np.concatenate([audio_out, silence, turn], axis=0)
                else:
                    audio_out = np.concatenate(audio_turns, axis=0)

                output = numpy_audio_to_comfy(audio_out, sample_rate)

            finally:
                # Always run on completion, cancellation, or error.
                if not keep_model_loaded:
                    unload_engine()
                elif offload_to_cpu:
                    offload_engine_to_cpu()

            return (output,)


# ---------------------------------------------------------------------------
# Shared helpers (used by both the v3 class method and v2 instance method)
# ---------------------------------------------------------------------------

def _get_engine(model_path, device, precision, attention, compile_model, keep_loaded=False, offload_to_cpu=False):
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


def _check_interrupt():
    if _MM:
        try:
            mm.throw_exception_if_processing_interrupted()
        except Exception:
            cancel_event.set()
            raise
