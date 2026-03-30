"""Fish Audio S2 - Multi-Speaker Split TTS node.

Outputs combined audio + separate per-speaker audio tracks.
Each speaker track contains their audio when speaking, silence when not.
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
from .multi_speaker_node import (
    _speaker_inputs,
    _parse_dialogue_lines,
    MAX_SPEAKERS,
    _get_engine,
    _check_interrupt,
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

try:
    from comfy_api.latest import IO
    _V3 = True
except ImportError:
    _V3 = False

logger = logging.getLogger("FishAudioS2")


if _V3:
    class FishS2MultiSpeakerSplitTTS(IO.ComfyNode):
        @classmethod
        def define_schema(cls) -> IO.Schema:
            model_names = get_model_names()

            speaker_options = [
                IO.DynamicCombo.Option(
                    key=str(n),
                    inputs=_speaker_inputs(n),
                )
                for n in range(2, MAX_SPEAKERS + 1)
            ]

            outputs = [IO.Audio.Output(display_name="audio")]
            for i in range(1, MAX_SPEAKERS + 1):
                outputs.append(IO.Audio.Output(display_name=f"speaker_{i}_audio"))

            return IO.Schema(
                node_id="FishS2MultiSpeakerSplitTTS",
                display_name="Fish S2 Multi-Speaker Split TTS",
                category="FishAudioS2",
                description=(
                    "Multi-speaker TTS with per-speaker audio outputs. "
                    "Outputs combined audio + separate track for each speaker. "
                    "Useful for multi-speaker lip sync workflows."
                ),
                inputs=[
                    IO.Combo.Input(
                        "model_path",
                        options=model_names,
                        tooltip="S2-Pro checkpoint folder name.",
                    ),
                    IO.String.Input(
                        "text",
                        multiline=True,
                        default=(
                            "[speaker_1]: Hello, I'm speaker one.\n"
                            "[speaker_2]: And I'm speaker two!"
                        ),
                        tooltip="Multi-speaker text. Use [speaker_1]:, [speaker_2]:, ...",
                    ),
                    IO.Combo.Input(
                        "language",
                        options=LANGUAGES,
                        tooltip="Language hint.",
                    ),
                    IO.Combo.Input(
                        "device",
                        options=["auto", "cuda", "cpu", "mps"],
                        tooltip="Compute device.",
                    ),
                    IO.Combo.Input(
                        "precision",
                        options=["auto", "bfloat16", "float16", "float32"],
                        tooltip="Model precision.",
                    ),
                    IO.Combo.Input(
                        "attention",
                        options=["auto", "sdpa", "sage_attention", "flash_attention"],
                        tooltip="Attention kernel.",
                    ),
                    IO.Int.Input(
                        "max_new_tokens",
                        default=0, min=0, max=4096, step=64,
                        tooltip="Max acoustic tokens. 0 = auto.",
                    ),
                    IO.Int.Input(
                        "chunk_length",
                        default=200, min=100, max=400, step=10,
                        tooltip="Chunk length for synthesis.",
                    ),
                    IO.Float.Input(
                        "temperature",
                        default=0.8, min=0.1, max=1.0, step=0.05,
                        tooltip="Sampling temperature.",
                    ),
                    IO.Float.Input(
                        "top_p",
                        default=0.8, min=0.1, max=1.0, step=0.05,
                        tooltip="Top-p nucleus sampling.",
                    ),
                    IO.Float.Input(
                        "repetition_penalty",
                        default=1.1, min=0.9, max=2.0, step=0.05,
                        tooltip="Repetition penalty.",
                    ),
                    IO.Int.Input(
                        "seed",
                        default=0, min=0, max=2**31 - 1,
                        tooltip="Random seed.",
                    ),
                    IO.Boolean.Input(
                        "keep_model_loaded",
                        default=True,
                        tooltip="Keep model in VRAM between runs.",
                    ),
                    IO.Boolean.Input(
                        "offload_to_cpu",
                        default=False,
                        tooltip="Offload to CPU after generation.",
                    ),
                    IO.Boolean.Input(
                        "compile_model",
                        default=False,
                        tooltip="torch.compile (not on Windows).",
                    ),
                    IO.Float.Input(
                        "pause_after_speaker",
                        default=0.4, min=0.0, max=2.0, step=0.1,
                        tooltip="Seconds of silence after each turn.",
                    ),
                    IO.DynamicCombo.Input(
                        "num_speakers",
                        options=speaker_options,
                        display_name="Number of Speakers",
                        tooltip=f"How many speakers (2-{MAX_SPEAKERS}).",
                    ),
                ],
                outputs=outputs,
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

            from fish_speech_s2.utils.schema import ServeReferenceAudio, ServeTTSRequest

            n = int(num_speakers["num_speakers"])

            references = {}
            missing = []
            for i in range(1, n + 1):
                speaker_audio = num_speakers.get(f"speaker_{i}_audio")
                speaker_ref_text = num_speakers.get(f"speaker_{i}_ref_text") or ""

                if speaker_audio is None:
                    missing.append(i)
                else:
                    logger.info(f"Encoding reference audio for speaker {i}...")
                    ref_bytes = audio_bytes_from_comfy(speaker_audio)
                    references[i - 1] = ServeReferenceAudio(
                        audio=ref_bytes,
                        text=speaker_ref_text.strip(),
                    )

            if missing:
                missing_str = ", ".join(f"speaker_{i}_audio" for i in missing)
                raise ValueError(f"Missing reference audio: {missing_str}")

            _check_interrupt()

            dialogue_lines = _parse_dialogue_lines(text)
            if not dialogue_lines:
                raise ValueError("No speaker lines found. Use [speaker_1]: text format.")

            logger.info(
                f"Multi-speaker Split TTS ({n} speakers): {len(dialogue_lines)} lines"
            )

            tokens = max_new_tokens if max_new_tokens > 0 else 0
            sample_rate = 44100

            speaker_tracks = {i: [] for i in range(MAX_SPEAKERS)}
            combined_parts = []

            pause_samples = int(pause_after_speaker * sample_rate) if pause_after_speaker > 0 else 0
            silence = np.zeros(pause_samples, dtype=np.float32) if pause_samples > 0 else None

            total_steps = len(dialogue_lines)
            pbar = ProgressBar(total_steps) if _PBAR else None

            try:
                for line_idx, (speaker_idx, line_text) in enumerate(dialogue_lines):
                    _check_interrupt()

                    if speaker_idx not in references:
                        raise ValueError(
                            f"Line {line_idx + 1} uses speaker {speaker_idx + 1} "
                            f"but no reference audio provided."
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
                            raise RuntimeError(f"Error on line {line_idx + 1}: {result.error}")
                        if result.code == "final":
                            sr, line_audio = result.audio
                            sample_rate = sr

                    if line_audio is None:
                        raise RuntimeError(f"No audio for line {line_idx + 1}.")

                    line_silence = np.zeros_like(line_audio)

                    for i in range(MAX_SPEAKERS):
                        if i == speaker_idx:
                            speaker_tracks[i].append(line_audio)
                            if silence is not None:
                                speaker_tracks[i].append(silence)
                        else:
                            speaker_tracks[i].append(line_silence)
                            if silence is not None:
                                speaker_tracks[i].append(silence)

                    combined_parts.append(line_audio)
                    if silence is not None:
                        combined_parts.append(silence)

                    if pbar:
                        pbar.update_absolute(line_idx + 1, total_steps)

                combined_audio = np.concatenate(combined_parts, axis=0)
                combined_out = numpy_audio_to_comfy(combined_audio, sample_rate)

                speaker_outputs = []
                for i in range(MAX_SPEAKERS):
                    if speaker_tracks[i]:
                        track_audio = np.concatenate(speaker_tracks[i], axis=0)
                    else:
                        track_audio = np.zeros(len(combined_audio), dtype=np.float32)
                    speaker_outputs.append(numpy_audio_to_comfy(track_audio, sample_rate))

            finally:
                if not keep_model_loaded:
                    unload_engine()
                elif offload_to_cpu:
                    offload_engine_to_cpu()

            return IO.NodeOutput(combined_out, *speaker_outputs)

else:
    class FishS2MultiSpeakerSplitTTS:
        @classmethod
        def INPUT_TYPES(cls):
            model_names = get_model_names()
            optional_inputs = {}
            for i in range(1, MAX_SPEAKERS + 1):
                optional_inputs[f"speaker_{i}_audio"] = ("AUDIO", {
                    "tooltip": f"Reference audio for speaker {i}.",
                })
                optional_inputs[f"speaker_{i}_ref_text"] = ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": f"Transcript for speaker {i} reference.",
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
                    }),
                    "language": (LANGUAGES, {"default": "auto"}),
                    "device": (["auto", "cuda", "cpu", "mps"], {"default": "auto"}),
                    "precision": (["auto", "bfloat16", "float16", "float32"], {"default": "auto"}),
                    "attention": (["auto", "sdpa", "sage_attention", "flash_attention"], {"default": "auto"}),
                    **COMMON_GENERATION_INPUTS,
                    "pause_after_speaker": ("FLOAT", {
                        "default": 0.4, "min": 0.0, "max": 5.0, "step": 0.1,
                    }),
                    "keep_model_loaded": ("BOOLEAN", {"default": True}),
                    "offload_to_cpu": ("BOOLEAN", {"default": False}),
                    "compile_model": ("BOOLEAN", {"default": False}),
                },
                "optional": optional_inputs,
            }

        RETURN_TYPES = ("AUDIO",) + ("AUDIO",) * MAX_SPEAKERS
        RETURN_NAMES = ("audio",) + tuple(f"speaker_{i}_audio" for i in range(1, MAX_SPEAKERS + 1))
        FUNCTION = "generate"
        CATEGORY = "FishAudioS2"
        DESCRIPTION = "Multi-speaker TTS with per-speaker audio outputs."

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

            from fish_speech_s2.utils.schema import ServeReferenceAudio, ServeTTSRequest

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
                    references[i - 1] = ServeReferenceAudio(
                        audio=ref_bytes, text=speaker_ref_text.strip()
                    )

            if missing:
                missing_str = ", ".join(f"speaker_{i}_audio" for i in missing)
                raise ValueError(f"Missing reference audio: {missing_str}")

            _check_interrupt()

            dialogue_lines = _parse_dialogue_lines(text)
            if not dialogue_lines:
                raise ValueError("No speaker lines found. Use [speaker_1]: text format.")

            logger.info(
                f"Multi-speaker Split TTS ({num_speakers} speakers): "
                f"{len(dialogue_lines)} lines"
            )

            tokens = max_new_tokens if max_new_tokens > 0 else 0
            sample_rate = 44100

            speaker_tracks = {i: [] for i in range(MAX_SPEAKERS)}
            combined_parts = []

            pause_samples = int(pause_after_speaker * sample_rate) if pause_after_speaker > 0 else 0
            silence = np.zeros(pause_samples, dtype=np.float32) if pause_samples > 0 else None

            total_steps = len(dialogue_lines)
            pbar = ProgressBar(total_steps) if _PBAR else None

            try:
                for line_idx, (speaker_idx, line_text) in enumerate(dialogue_lines):
                    _check_interrupt()

                    if speaker_idx not in references:
                        raise ValueError(
                            f"Line {line_idx + 1} uses speaker {speaker_idx + 1} "
                            f"but no reference audio provided."
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
                            raise RuntimeError(f"Error on line {line_idx + 1}: {result.error}")
                        if result.code == "final":
                            sr, line_audio = result.audio
                            sample_rate = sr

                    if line_audio is None:
                        raise RuntimeError(f"No audio for line {line_idx + 1}.")

                    line_silence = np.zeros_like(line_audio)

                    for i in range(MAX_SPEAKERS):
                        if i == speaker_idx:
                            speaker_tracks[i].append(line_audio)
                            if silence is not None:
                                speaker_tracks[i].append(silence)
                        else:
                            speaker_tracks[i].append(line_silence)
                            if silence is not None:
                                speaker_tracks[i].append(silence)

                    combined_parts.append(line_audio)
                    if silence is not None:
                        combined_parts.append(silence)

                    if pbar:
                        pbar.update_absolute(line_idx + 1, total_steps)

                combined_audio = np.concatenate(combined_parts, axis=0)
                combined_out = numpy_audio_to_comfy(combined_audio, sample_rate)

                speaker_outputs = []
                for i in range(MAX_SPEAKERS):
                    if speaker_tracks[i]:
                        track_audio = np.concatenate(speaker_tracks[i], axis=0)
                    else:
                        track_audio = np.zeros(len(combined_audio), dtype=np.float32)
                    speaker_outputs.append(numpy_audio_to_comfy(track_audio, sample_rate))

            finally:
                if not keep_model_loaded:
                    unload_engine()
                elif offload_to_cpu:
                    offload_engine_to_cpu()

            return (combined_out,) + tuple(speaker_outputs)
