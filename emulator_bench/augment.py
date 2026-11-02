"""Shuffles of the reactants and the products of a reaction (CLAIRE data augmentation).

The released train_augmented.csv holds every distinct ordering when a reaction has at most
MAX_ROWS of them, else MAX_ROWS orderings drawn with replacement. The augmentation code is not released.
Here the draws are seeded by the reaction, so every run and every host gets the same rows.
"""

from __future__ import annotations

import itertools
import math
import random
import zlib

MAX_ROWS = 10


def augment(rxn: str) -> list[str]:
    """The reaction first, then its other orderings, `MAX_ROWS` rows at most."""
    left, right = (side.split(".") for side in rxn.split(">>"))
    if math.factorial(len(left)) * math.factorial(len(right)) <= MAX_ROWS:
        orderings = dict.fromkeys(itertools.product(itertools.permutations(left), itertools.permutations(right)))
    else:
        rng = random.Random(zlib.crc32(rxn.encode()))
        draws = [(tuple(rng.sample(left, len(left))), tuple(rng.sample(right, len(right)))) for _ in range(MAX_ROWS - 1)]
        orderings = [(tuple(left), tuple(right)), *draws]
    return [".".join(a) + ">>" + ".".join(b) for a, b in orderings]


def canonical(rxn: str) -> str:
    """The reaction with sorted reactants and products. DRFP is the same for every ordering."""
    left, right = (".".join(sorted(side.split("."))) for side in rxn.split(">>"))
    return left + ">>" + right
