"""Shared prompt and causal frame input construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch


DEFAULT_SYSTEM_PROMPT = "You are a concise spoken-dialogue assistant."


def prompt_token_ids(tokenizer: object, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> list[int]:
    template = getattr(tokenizer, "apply_chat_template", None)
    if callable(template):
        values = template([{"role": "system", "content": system_prompt}], tokenize=True,
                          add_generation_prompt=True)
        if isinstance(values, Mapping):
            values = values["input_ids"]
        if isinstance(values, torch.Tensor):
            values = values.tolist()
        if values and isinstance(values[0], (tuple, list)):
            values = values[0]
        return [int(value) for value in values]
    return list(tokenizer.encode(system_prompt, add_special_tokens=True))


def frame_event_inputs(previous_event_id: int | None, *, bos_token_id: int,
                       control_ids: Sequence[int]) -> tuple[int, bool, int, bool]:
    """Return text ID/mask and control ID/mask for one causal frame."""
    if previous_event_id is None:
        return bos_token_id, True, 0, False
    if previous_event_id in control_ids:
        return 0, False, previous_event_id, True
    return previous_event_id, True, 0, False
