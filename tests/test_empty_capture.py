"""Empty-capture decline — zero recorded kernels is a capture FAILURE.

``torch.cuda.graph`` warns "The CUDA Graph is empty ..."
(aten/src/ATen/cuda/CUDAGraph.cpp) when a capture records no kernels;
arming that graph would replay stale memory while
``stats["cuda_graph"]`` still reads True (and ``--no-verify``
delivery would ship it).  ``_GraphReplayModule.capture`` therefore
raises on the warning — the same failure path every other capture
error takes: ``cuda_graph_failure`` recorded, eager fallback — never
an armed no-op graph.  The CUDA record path itself stays
``requires_cuda``; these tests pin the detection helper directly on
CPU with fabricated warning objects.
"""

from __future__ import annotations

import warnings

from kit_cuda.runners import (
    _empty_capture_warning,
    _reemit_capture_warnings,
)


def _warnmsg(text: str) -> warnings.WarningMessage:
    return warnings.WarningMessage(
        message=text, category=UserWarning, filename="f.py", lineno=1
    )


def test_empty_capture_warning_detects_torch_message():
    """Torch's verbatim warning (aten CUDAGraph.cpp) is detected."""
    ws = [
        _warnmsg(
            "The CUDA Graph is empty. This usually means that the "
            "graph was attempted to be captured on wrong device or "
            "stream."
        )
    ]
    assert _empty_capture_warning(ws) is ws[0]


def test_empty_capture_warning_variants_case_insensitive():
    """The older "CUDA Graph is empty" phrasing and case variants
    match; unrelated warnings do not."""
    assert (
        _empty_capture_warning(
            [_warnmsg("UserWarning: CUDA GRAPH IS EMPTY")]
        )
        is not None
    )
    assert _empty_capture_warning([_warnmsg("an empty graph ahead")])
    assert (
        _empty_capture_warning([_warnmsg("some other warning")]) is None
    )
    assert _empty_capture_warning([]) is None


def test_reemit_capture_warnings_skips_only_the_trigger():
    """catch_warnings(record=True) must not swallow unrelated
    warnings: everything except the matched empty-capture warning is
    re-emitted verbatim."""
    empty = _warnmsg("The CUDA Graph is empty ...")
    other = _warnmsg("unrelated capture warning")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _reemit_capture_warnings([empty, other], empty)
    assert len(caught) == 1
    # warn_explicit wraps the str in the category — compare text
    assert str(caught[0].message) == other.message
    # skipping None re-emits everything
    with warnings.catch_warnings(record=True) as caught2:
        warnings.simplefilter("always")
        _reemit_capture_warnings([empty, other], None)
    assert len(caught2) == 2
