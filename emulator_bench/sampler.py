"""Exact fast replacement for the original triplet sampling in dev/training/train-pred_rxn_EC.py.

The original draws from Python's `random` module. This module reads the same Mersenne Twister word
stream and applies the same arithmetic as random.choice and random.choices, so every draw is
identical. numba runs the loop.
"""

from __future__ import annotations

import random

import numpy as np
from numba import njit

MARGIN_WORDS = 1 << 16


class MTStream:
    """The 32-bit output words of random.seed(seed), consumed in order."""

    def __init__(self, seed: int):
        random.seed(seed)
        state = random.getstate()[1]
        self.bit_generator = np.random.MT19937()
        self.bit_generator.state = {
            "bit_generator": "MT19937",
            "state": {"key": np.array(state[:624], dtype=np.uint32), "pos": state[624]},
        }
        self.words = np.empty(0, dtype=np.uint32)
        self.pos = 0

    def window(self, need: int) -> np.ndarray:
        """Unread words, at least `need` of them."""
        left = len(self.words) - self.pos
        if left < need:
            fresh = self.bit_generator.random_raw(max(need, 1 << 22)).astype(np.uint32)
            self.words = np.concatenate([self.words[self.pos :], fresh])
            self.pos = 0
        return self.words[self.pos :]

    def advance(self, used: int) -> None:
        self.pos += used

    def state(self) -> dict:
        return {"bit_generator": self.bit_generator.state, "words": self.words[self.pos :].copy()}

    def load_state(self, state: dict) -> None:
        self.bit_generator.state = state["bit_generator"]
        self.words, self.pos = state["words"], 0


@njit(cache=True)
def _below(words, p, n):
    """random.randbelow(n) for n >= 1: returns (value, next position)."""
    k = 0
    m = n
    while m > 0:
        k += 1
        m >>= 1
    while True:
        r = np.int64(words[p]) >> (32 - k)
        p += 1
        if r < n:
            return r, p


@njit(cache=True)
def _random(words, p):
    a = np.int64(words[p]) >> 5
    b = np.int64(words[p + 1]) >> 6
    return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0), p + 2


@njit(cache=True)
def sample_triplets(words, order, row_ec, ec_ptr, ec_rows, out):
    """Triplets (anchor row, positive row, negative row) for the anchor rows in `order`.

    Per row, the original draws the positive with random.choice over the rows of the anchor EC, then
    the negative EC with random.choices over all ECs until it differs, then a row of that EC.
    Returns the number of triplets done and the number of words used. Stops early when fewer than
    MARGIN_WORDS words are left; the caller then refills the stream and continues.
    """
    p = 0
    n_ecs = len(ec_ptr) - 1
    for t in range(len(order)):
        if p + MARGIN_WORDS > len(words):
            return t, p
        row = order[t]
        ec = row_ec[row]
        n, p = _below(words, p, ec_ptr[ec + 1] - ec_ptr[ec])
        positive = ec_rows[ec_ptr[ec] + n]
        neg = ec
        while neg == ec:
            x, p = _random(words, p)
            neg = np.int64(x * n_ecs)
        n, p = _below(words, p, ec_ptr[neg + 1] - ec_ptr[neg])
        out[t, 0] = row
        out[t, 1] = positive
        out[t, 2] = ec_rows[ec_ptr[neg] + n]
    return len(order), p
