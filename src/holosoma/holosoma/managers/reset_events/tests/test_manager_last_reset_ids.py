"""ResetEventManager.last_reset_ids contract.

Consumers (e.g. wbt-force-v2 `force/term_rate_{active,inactive}_ext`) rely on
the manager caching the env_ids passed to the most recent ``reset_scene``.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from holosoma.managers.reset_events.manager import ResetEventManager


def _empty_manager() -> ResetEventManager:
    mgr = ResetEventManager.__new__(ResetEventManager)
    mgr.events = []
    mgr.last_reset_ids = None
    return mgr


def test_last_reset_ids_is_none_before_first_reset() -> None:
    mgr = _empty_manager()
    assert mgr.last_reset_ids is None


def test_last_reset_ids_reflects_latest_reset() -> None:
    mgr = _empty_manager()
    ids_a = torch.tensor([0, 2, 5], dtype=torch.long)
    mgr.reset_scene(ids_a)
    assert torch.equal(mgr.last_reset_ids, ids_a)

    ids_b = torch.tensor([1], dtype=torch.long)
    mgr.reset_scene(ids_b)
    assert torch.equal(mgr.last_reset_ids, ids_b)


def test_last_reset_ids_forwarded_to_each_event() -> None:
    mgr = _empty_manager()
    seen: list[torch.Tensor] = []
    mgr.events = [SimpleNamespace(reset=lambda env_ids: seen.append(env_ids.clone()))]  # type: ignore[list-item]
    ids = torch.tensor([3, 4], dtype=torch.long)
    mgr.reset_scene(ids)
    assert len(seen) == 1
    assert torch.equal(seen[0], ids)
    assert torch.equal(mgr.last_reset_ids, ids)
