"""Shared model cache for Fish Audio S2 nodes."""

import gc
import logging
from typing import Any

import torch

logger = logging.getLogger("FishAudioS2")

# Module-level cache: keyed by (model_path, device, precision)
_cached_engine: Any = None
_cached_key: tuple = ()

# Set to True by the node when keep_model_loaded=True, so the soft_empty_cache
# hook knows not to evict the engine under automatic memory pressure.
_keep_loaded: bool = False


def get_cache_key(model_path: str, device: str, precision: str, attention: str) -> tuple:
    return (model_path, device, precision, attention)


def get_cached_engine():
    return _cached_engine, _cached_key


def set_cached_engine(engine: Any, key: tuple, keep_loaded: bool = False):
    global _cached_engine, _cached_key, _keep_loaded
    _cached_engine = engine
    _cached_key = key
    _keep_loaded = keep_loaded


def unload_engine():
    global _cached_engine, _cached_key, _keep_loaded
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
