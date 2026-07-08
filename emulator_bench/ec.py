import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


LEVELS = {"l1": 1, "l2": 2, "l3": 3, "l4": 4}
EC_RE = re.compile(r"^(?P<a>[1-7])(?:\.(?P<b>\d+|-|x))?(?:\.(?P<c>\d+|-|x))?(?:\.(?P<d>\d+|n\d*|-|x))?$", re.I)


@dataclass(frozen=True)
class ParsedEC:
    parts: Tuple[str, str, str, str]
    provisional: bool = False
    partial: bool = False


def normalize_ec(value: str) -> str:
    text = str(value).strip()
    if text.lower().startswith("ec:"):
        text = text.split(":", 1)[1].strip()
    return text.replace(" ", "")


def parse_ec(value: str, allow_provisional: bool = False) -> ParsedEC:
    text = normalize_ec(value)
    match = EC_RE.match(text)
    if not match:
        raise ValueError(f"Invalid EC number syntax: {value}")
    raw = [match.group(name) for name in ("a", "b", "c", "d")]
    parts = [raw[0]]
    for index in range(1, 4):
        part = raw[index]
        parts.append("-" if part is None else part.lower())

    seen_unspecified = False
    for index, part in enumerate(parts):
        unspecified = part in {"-", "x"}
        if seen_unspecified and not unspecified:
            raise ValueError(f"Invalid EC number with specified child after wildcard: {value}")
        seen_unspecified = seen_unspecified or unspecified
        if part.startswith("n") and index != 3:
            raise ValueError(f"Provisional EC marker is only valid in L4: {value}")

    provisional = parts[3].startswith("n")
    if provisional and not allow_provisional:
        raise ValueError(f"Provisional EC labels are disabled by default: {value}")
    return ParsedEC(tuple(parts), provisional=provisional, partial=any(part in {"-", "x"} for part in parts))


def is_unlabeled_ec(value: str) -> bool:
    return normalize_ec(value).lower() in {"-.-.-.-", "x.x.x.x", "", "nan", "none", "null"}


def ec_label(value: str, level: str, allow_provisional: bool = False) -> str:
    if level not in LEVELS:
        raise ValueError(f"Unsupported EC level: {level}")
    parsed = parse_ec(value, allow_provisional=allow_provisional)
    depth = LEVELS[level]
    parts = parsed.parts[:depth]
    if any(part in {"-", "x"} for part in parts):
        raise ValueError(f"Cannot build {level} label from partial EC {value}")
    return ".".join(parts)


def build_level_vocab(values: Iterable[str], level: str, allow_provisional: bool = False) -> Set[str]:
    vocab: Set[str] = set()
    for value in values:
        try:
            vocab.add(ec_label(value, level, allow_provisional=allow_provisional))
        except ValueError:
            continue
    return vocab


def is_allowed_label(label: str, vocab: Set[str]) -> bool:
    return label in vocab


def partial_match(query: str, candidate: str, vocab: Optional[Set[str]] = None, allow_provisional: bool = False) -> bool:
    query_parts = parse_ec(query, allow_provisional=allow_provisional).parts
    candidate_parts = parse_ec(candidate, allow_provisional=allow_provisional).parts
    for q_part, c_part in zip(query_parts, candidate_parts):
        if q_part in {"-", "x"}:
            break
        if q_part != c_part:
            return False
    if vocab is not None:
        return ".".join([part for part in candidate_parts if part not in {"-", "x"}]) in vocab or candidate in vocab
    return True


def prefix_accuracy(y_true: Sequence[str], y_pred: Sequence[str], level: str) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal length")
    if not y_true:
        return 0.0
    correct = 0
    for true, pred in zip(y_true, y_pred):
        try:
            correct += int(ec_label(true, level) == ec_label(pred, level))
        except ValueError:
            correct += 0
    return correct / len(y_true)
