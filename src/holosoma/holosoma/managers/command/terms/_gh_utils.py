"""Small tensor-time-lerp + sampling helpers reused by WBT wrist-force.

Adapted from gentle-humanoid-training (third_party/gentle-humanoid-training/
active_adaptation/envs/mdp/commands/utils.py and admittance.py, commit
``9fd915d9639fc9b219e4b12b04076f67db82c7b5``).

Copied (not imported) into Holosoma so we keep a zero-runtime dependency on
``third_party/`` — the original GH code only uses ``torch`` + ``typing`` so
this file stands alone.

The three primitives in this module:

* :class:`TemporalLerp` — a compact state machine that ramps a per-element
  tensor from ``start`` to ``end`` over an integer number of steps.
* :func:`clamp_norm` — clamps a vector's L2 norm to at most ``max`` along a
  given dimension without changing its direction.
* :func:`random_uniform` — batched ``torch.rand`` scaled to a ``[min, max]``
  range on the requested device.

``AdmittanceMassChain`` is intentionally **not** ported: WBT wrist-force does
not need an admittance body-chain, and omitting it keeps the surface small.
"""

from __future__ import annotations

import torch

__all__ = ["TemporalLerp", "clamp_norm", "random_uniform"]

_CPU_DEVICE = torch.device("cpu")


class TemporalLerp:
    """Manage time-varying tensors with start/end/total_steps.

    Works on arbitrary tensor shapes like ``(N, M, D)``; time is discrete
    (int steps). The last broadcast dim is handled automatically so we can
    share a clock across ``D`` in ``(..., D)``.
    """

    def __init__(
        self,
        shape: tuple[int, ...] | list[int] | torch.Size,
        device: torch.device,
        default: float = 0.0,
        easing: str = "linear",  # "linear" | "smoothstep"
        clamp: tuple[float, float] | None = None,
    ) -> None:
        self.device = device
        self.value = torch.full(shape, default, dtype=torch.float32, device=device)
        self._start = self.value.clone()
        self._end = self.value.clone()

        # time states broadcast over the last dim
        tshape = self.value.shape[:1] + (1,) * (len(self.value.shape) - 1)
        self._t = torch.zeros(tshape, dtype=torch.int32, device=device)
        # initial total_steps=1 avoids a /0 before any `set` call.
        self._T = torch.ones(tshape, dtype=torch.int32, device=device)
        self._active = torch.zeros(tshape, dtype=torch.bool, device=device)
        self.easing = easing
        self.clamp = clamp

    # ---------- public API ----------
    @torch.no_grad()
    def set(
        self,
        env_ids: torch.Tensor | None,
        end: torch.Tensor | float | None = None,
        delta: torch.Tensor | float | None = None,
        total_steps: torch.Tensor | int = 0,
        start: torch.Tensor | None = None,
    ) -> None:
        """Start (or override) a ramp for a subset of the first-dim indices.

        * If ``start`` is ``None``, the current value at those indices is
          reused as the start.
        * Either ``end`` or ``delta`` must be provided (not both).
        * ``total_steps`` may be an int or a tensor broadcastable to the
          clock shape ``(..., 1)``.
        """
        if end is None and delta is None:
            raise ValueError("Either 'end' or 'delta' must be provided.")
        if end is not None and delta is not None:
            raise ValueError("Only one of 'end' or 'delta' can be provided.")

        index = (slice(None),) if env_ids is None else (env_ids,)
        cur_start = self.value[index] if start is None else start
        self._start[index] = cur_start
        if end is not None:
            self._end[index] = end
        elif delta is not None:
            self._end[index] = cur_start + delta
        self._t[index] = 0
        if isinstance(total_steps, int):
            self._T[index] = total_steps
        elif isinstance(total_steps, torch.Tensor):
            self._T[index] = total_steps.view((-1,) + (1,) * (len(self.value.shape) - 1))
        self._active[index] = torch.ones_like(self._active[index], dtype=torch.bool)
        self.value[index] = cur_start

    @torch.no_grad()
    def update_time(self, steps: int = 1) -> None:
        """Advance the clock for all active elements and refresh value."""
        if not self._active.any():
            return
        self._t[self._active] += int(steps)
        self._update_value()
        done = self._t >= self._T
        self._active = self._active & (~done)

    @torch.no_grad()
    def reset(
        self,
        env_ids: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
    ) -> None:
        index = (slice(None),) if env_ids is None else (env_ids,)
        if value is not None:
            self.value[index] = value
        self._start[index] = self.value[index]
        self._end[index] = self.value[index]
        self._t[index].zero_()
        self._T[index].fill_(1)
        self._active[index].zero_()

    @property
    def current(self) -> torch.Tensor:
        return self.value

    @property
    def mask_active(self) -> torch.Tensor:
        return self._active.view(-1)

    @property
    def mask_done(self) -> torch.Tensor:
        return (~self._active).view(-1)

    @property
    def time_left(self) -> torch.Tensor:
        return (self._T - self._t).clamp_min(0).view(-1)

    # ---------- internals ----------
    def _ease(self, a: torch.Tensor) -> torch.Tensor:
        if self.easing == "linear":
            return a
        if self.easing == "smoothstep":
            # 3a^2 - 2a^3
            return a * a * (3 - 2 * a)
        raise ValueError(f"Unknown easing function: {self.easing!r}. Use 'linear' or 'smoothstep'.")

    def _update_value(self) -> None:
        self.value = self._ease((self._t / self._T).clamp_(0.0, 1.0)) * (self._end - self._start) + self._start
        if self.clamp is not None:
            self.value.clamp_(min=self.clamp[0], max=self.clamp[1])


def clamp_norm(
    x: torch.Tensor,
    max_norm: float = 100.0,
    dim: int = -1,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Clamp ``x`` so that ``x.norm(dim=dim) <= max_norm`` without rotating it."""
    x_norm = x.norm(dim=dim, keepdim=True).clamp_min(eps)
    max_t = torch.as_tensor(max_norm, device=x.device, dtype=x.dtype)
    scale = (max_t / x_norm).clamp(max=1.0)
    return x * scale


def random_uniform(
    shape: int | tuple[int, ...],
    low: float = 0.0,
    high: float = 0.0,
    device: torch.device = _CPU_DEVICE,
) -> torch.Tensor:
    """Sample ``[low, high)`` uniformly with ``torch.rand`` on ``device``."""
    return torch.rand(shape, device=device, dtype=torch.float32) * (high - low) + low
