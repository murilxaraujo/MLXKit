"""Deterministic shuffle algorithm shared with MLXKit's DataLoader."""

from __future__ import annotations


def splitmix64_shuffle_indices(count: int, seed: int) -> list[int]:
    """Return a Fisher–Yates order compatible with MLXKit DataLoader."""
    if count < 0 or seed < 0:
        raise ValueError("count and seed cannot be negative")
    mask = (1 << 64) - 1
    state = seed & mask
    order = list(range(count))
    for index in range(count - 1, 0, -1):
        state = (state + 0x9E3779B97F4A7C15) & mask
        value = state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
        value ^= value >> 31
        swap_index = value % (index + 1)
        order[index], order[swap_index] = order[swap_index], order[index]
    return order
