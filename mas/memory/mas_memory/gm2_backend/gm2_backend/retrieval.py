from __future__ import annotations

import re

from ...common import MASMessage


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(text or "").lower()))


def _score_query_overlap(query: str, message: MASMessage) -> float:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return 0.0
    hay = " ".join(
        [
            str(message.task_main or ""),
            str(message.task_description or ""),
            str(message.task_trajectory or ""),
        ]
    )
    hay_tokens = _tokenize(hay)
    if not hay_tokens:
        return 0.0
    overlap = len(query_tokens & hay_tokens)
    return overlap / max(len(query_tokens), 1)


def rank_messages_for_query(
    query: str,
    messages: list[MASMessage],
    *,
    topk: int = 3,
    label: bool | None = None,
) -> list[MASMessage]:
    filtered = [
        message
        for message in messages
        if label is None or bool(message.label) is bool(label)
    ]
    ranked = sorted(
        filtered,
        key=lambda message: (
            _score_query_overlap(query, message),
            len(str(message.task_trajectory or "")),
        ),
        reverse=True,
    )
    return ranked[: max(int(topk), 0)]
