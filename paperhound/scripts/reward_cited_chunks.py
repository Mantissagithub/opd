import json
import re
import string
from collections.abc import Iterable


def _normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _as_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        return _as_list(parsed)
    if isinstance(value, dict):
        chunks = value.get("cited_chunks", [])
        return _as_list(chunks)
    if isinstance(value, Iterable):
        return [str(item) for item in value]
    return []


def _extract_predicted_chunks(solution_str: str) -> tuple[list[str], bool]:
    text = solution_str.strip()
    candidates = [text]
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidates.insert(0, match.group(0))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        return _as_list(parsed), isinstance(parsed, dict) and "cited_chunks" in parsed

    quoted = re.findall(r'"([^"]{24,})"', text)
    if quoted:
        return quoted, False
    return [line.strip("-* \t") for line in text.splitlines() if len(line.strip()) > 24], False


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    gold = {_normalize(chunk) for chunk in _as_list(ground_truth) if _normalize(chunk)}
    predicted_chunks, valid_json = _extract_predicted_chunks(solution_str)
    pred = {_normalize(chunk) for chunk in predicted_chunks if _normalize(chunk)}

    if not gold or not pred:
        return 0.05 if valid_json else 0.0

    hits = len(gold & pred)
    recall = hits / len(gold)
    precision = hits / len(pred)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    format_bonus = 0.1 if valid_json else 0.0
    return min(1.0, 0.55 * recall + 0.35 * f1 + format_bonus)


if __name__ == "__main__":
    gold = json.dumps(["chunk a", "chunk b"])
    good = '{"cited_chunks": ["chunk a", "chunk b"]}'
    partial = '{"cited_chunks": ["chunk a"]}'
    assert compute_score("x", good, gold) == 1.0
    assert 0.0 < compute_score("x", partial, gold) < 1.0
    assert compute_score("x", "not json", gold) == 0.0
    print("reward self-test passed")
