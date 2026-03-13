"""Shared model cache for Fish Audio S2 nodes."""

import gc
import logging
import queue
import threading
from typing import Any

import torch

logger = logging.getLogger("FishAudioS2")

# Module-level cache: keyed by (model_path, device, precision)
_cached_engine: Any = None
_cached_key: tuple = ()

# Set to True by the node when keep_model_loaded=True, so the soft_empty_cache
# hook knows not to evict the engine under automatic memory pressure.
_keep_loaded: bool = False

# Tracks whether the engine is currently offloaded to CPU.
_offloaded: bool = False

# Cancel event — set by the main thread when generation is interrupted.
# The worker thread checks this on each token and stops early.
cancel_event: threading.Event = threading.Event()


def get_cache_key(model_path: str, device: str, precision: str, attention: str, model_name: str = "") -> tuple:
    # Include model_name in key so s2-pro and s2-pro-bnb-nf4 are cached separately
    # (they share the same path but have different quantization)
    return (model_path, device, precision, attention, model_name)


def get_cached_engine():
    return _cached_engine, _cached_key


def set_cached_engine(engine: Any, key: tuple, keep_loaded: bool = False):
    global _cached_engine, _cached_key, _keep_loaded, _offloaded
    _cached_engine = engine
    _cached_key = key
    _keep_loaded = keep_loaded
    _offloaded = False


def is_offloaded() -> bool:
    return _offloaded


def offload_engine_to_cpu() -> None:
    """
    Move both the LLaMA model (inside the worker thread) and the decoder model
    to CPU, freeing VRAM while keeping the engine alive for the next run.

    _offloaded is only set True when BOTH components confirm success.
    If either fails the state stays consistent — no partial offload.

    When called after a cancellation the LLaMA worker may still be finishing
    its current job. We drain that job's output first so the worker becomes
    free to process our offload message.
    """
    global _offloaded

    if _cached_engine is None:
        return
    if _offloaded:
        logger.debug("Engine already offloaded to CPU — skipping.")
        return

    engine = _cached_engine
    decoder_ok = False
    llama_ok = False

    # --- Move decoder model to CPU immediately (main thread, no queue needed) ---
    try:
        engine.decoder_model.to("cpu")
        decoder_ok = True
        logger.info("Decoder model offloaded to CPU.")
    except Exception as e:
        logger.warning(f"Failed to offload decoder model: {e}")

    # --- Ask the LLaMA worker thread to move the model to CPU ---
    # The worker may still be finishing a cancelled generation — it will process
    # our offload message as soon as it finishes that job. We use a long timeout
    # to cover the worst case (long generation cancelled mid-way).
    try:
        from fish_speech.models.text2semantic.inference import GenerateRequest

        offload_response: queue.Queue = queue.Queue()
        engine.llama_queue.put(
            GenerateRequest(
                request={"__offload__": "cpu"},
                response_queue=offload_response,
            )
        )
        # 120s: enough for even a long generation to finish before we give up.
        try:
            result = offload_response.get(timeout=120)
            if getattr(result, "status", None) == "error":
                logger.warning(f"LLaMA offload reported error: {result.response}")
            else:
                llama_ok = True
                logger.info("LLaMA model offloaded to CPU.")
        except queue.Empty:
            logger.warning(
                "LLaMA offload timed out after 120s — VRAM not freed. "
                "The worker thread may be stuck. Restart ComfyUI to recover."
            )
    except Exception as e:
        logger.warning(f"Failed to send offload request to LLaMA worker: {e}")

    if decoder_ok and llama_ok:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        _offloaded = True
        logger.info("Engine offloaded to CPU. VRAM freed.")
    else:
        # Partial failure — roll back decoder to CUDA so state stays consistent.
        if decoder_ok and not llama_ok:
            try:
                engine.decoder_model.to("cuda")
                logger.warning("Offload rolled back — decoder moved back to CUDA.")
            except Exception as e:
                logger.warning(f"Rollback failed: {e}")
        logger.warning(
            "CPU offload failed — model remains in VRAM. "
            "Try again or restart ComfyUI."
        )


def resume_engine_to_cuda(device: str = "cuda") -> None:
    """
    Move the engine back from CPU to the target device before the next inference.
    """
    global _offloaded

    if _cached_engine is None:
        return
    if not _offloaded:
        return

    engine = _cached_engine

    # --- Move decoder model back to device ---
    try:
        engine.decoder_model.to(device)
        logger.info(f"Decoder model resumed to {device}.")
    except Exception as e:
        logger.warning(f"Failed to resume decoder model: {e}")

    # --- Ask the LLaMA worker thread to move back to device ---
    try:
        from fish_speech.models.text2semantic.inference import GenerateRequest

        response_queue: queue.Queue = queue.Queue()
        engine.llama_queue.put(
            GenerateRequest(
                request={"__offload__": device},
                response_queue=response_queue,
            )
        )
        try:
            result = response_queue.get(timeout=120)
            if getattr(result, "status", None) == "error":
                logger.warning(f"LLaMA resume reported error: {result.response}")
            else:
                logger.info(f"LLaMA model resumed to {device}.")
        except queue.Empty:
            logger.warning(
                "LLaMA resume timed out after 120s — model may still be on CPU. "
                "Restart ComfyUI to recover."
            )
    except Exception as e:
        logger.warning(f"Failed to send resume request to LLaMA worker: {e}")

    _offloaded = False


def unload_engine():
    global _cached_engine, _cached_key, _keep_loaded, _offloaded
    if _cached_engine is not None:
        logger.info("Unloading Fish S2 model from memory...")
        try:
            engine = _cached_engine
            if hasattr(engine, "llama_queue"):
                engine.llama_queue.put(None)  # sentinel to stop thread
        except Exception:
            pass
        del _cached_engine
        _cached_engine = None
        _cached_key = ()
        _keep_loaded = False
        _offloaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("Model unloaded and VRAM freed.")


def _hook_comfy_model_management():
    """
    Patch comfy.model_management so that ComfyUI's native 'Unload Models'
    button also clears our engine cache — but only when keep_model_loaded
    is False. If the user opted into keeping the model loaded, automatic
    memory pressure calls to soft_empty_cache will not evict our engine.
    """
    try:
        import comfy.model_management as mm
        _original = mm.soft_empty_cache

        def _patched_soft_empty_cache(*args, **kwargs):
            if not _keep_loaded:
                unload_engine()
            return _original(*args, **kwargs)

        mm.soft_empty_cache = _patched_soft_empty_cache
        logger.debug("Hooked comfy.model_management.soft_empty_cache for Fish S2 unload.")
    except Exception:
        pass  # not inside ComfyUI — no-op


# Hook at import time so it's active as soon as the node package loads.
_hook_comfy_model_management()
