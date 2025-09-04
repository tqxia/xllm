from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class ForwardContext:
    is_prefill: bool = False
    cu_seqlens: List[int] = None
    context_lens: List[int] = None


_forward_context: Optional[ForwardContext] = None


def get_forward_context() -> ForwardContext:
    global _forward_context
    assert _forward_context is not None
    return _forward_context

def set_forward_context(
    is_prefill: bool,
    cu_seqlens: List[int] = None,
    context_lens: List[int] = None,
) -> None:
    global _forward_context
    _forward_context = ForwardContext(is_prefill=is_prefill, cu_seqlens=cu_seqlens, context_lens=context_lens)
