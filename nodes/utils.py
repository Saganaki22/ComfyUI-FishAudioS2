"""Text chunking utilities for long-form TTS."""

import re
from typing import List


def count_words(text: str) -> int:
    """Approximate word count (CJK chars count as one word each)."""
    if not text or not text.strip():
        return 0
    cjk = len(re.findall(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", text))
    non_cjk = re.sub(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", " ", text)
    western = len(non_cjk.split())
    return cjk + western


def split_text_into_chunks(
    text: str,
    max_words_per_chunk: int = 200,
    per_batch_prefix: str = "",
) -> List[str]:
    """
    Split text into chunks suitable for TTS, respecting sentence boundaries.
    Prepends <|speaker:0|> tag to each chunk for native batching.
    If per_batch_prefix is provided, prepends it before the speaker tag.
    """
    text = text.strip()
    if not text:
        return []

    sentence_end = r"[.!?。！？]\s*|\n+"
    raw_sentences = re.split(f"({sentence_end})", text)
    sentences = []
    buf = ""
    for i, part in enumerate(raw_sentences):
        if re.match(sentence_end, part):
            buf += part
            if buf.strip():
                sentences.append(buf.strip())
            buf = ""
        else:
            buf = part
    if buf.strip():
        sentences.append(buf.strip())

    # Build prefix: speaker tag + user's custom prefix (inside the speech content)
    prefix = f"<|speaker:0|>{per_batch_prefix}"

    if not sentences:
        return [f"{prefix}{text}"] if text else []

    chunks = []
    current = []
    current_words = 0

    for sent in sentences:
        sent_words = count_words(sent)
        if sent_words > max_words_per_chunk:
            if current:
                chunk_text = " ".join(current)
                chunks.append(f"{prefix}{chunk_text}")
                current = []
                current_words = 0
            chunks.append(f"{prefix}{sent}")
            continue

        if current_words + sent_words > max_words_per_chunk and current:
            chunk_text = " ".join(current)
            chunks.append(f"{prefix}{chunk_text}")
            current = []
            current_words = 0

        current.append(sent)
        current_words += sent_words

    if current:
        chunk_text = " ".join(current)
        chunks.append(f"{prefix}{chunk_text}")

    return [c for c in chunks if c]
