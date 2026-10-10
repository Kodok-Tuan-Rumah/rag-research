"""Deterministic split of retrieval eval dataset into train/holdout."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATASET = ROOT / "data" / "evaluation" / "retrieval_eval_dataset.json"
TRAIN = ROOT / "data" / "evaluation" / "retrieval_train_eval.json"
HOLDOUT = ROOT / "data" / "evaluation" / "retrieval_holdout_eval.json"
SEED = 42


def _stratum_key(item: dict) -> tuple[str, str]:
    return item.get("question_type", "general"), item.get("difficulty", "medium")


def main() -> None:
    data = json.loads(DATASET.read_text())
    rng = random.Random(SEED)

    buckets: dict[tuple[str, str], list[dict]] = {}
    for item in data:
        buckets.setdefault(_stratum_key(item), []).append(item)

    train: list[dict] = []
    holdout: list[dict] = []
    for _key, items in sorted(buckets.items()):
        rng.shuffle(items)
        for i, item in enumerate(items):
            if i % 3 == 0:
                holdout.append(item)
            else:
                train.append(item)

    # low-count guard: ensure at least two holdout items if total allows
    if len(holdout) < 2 and len(data) >= 6:
        take = data[:2]
        for it in take:
            if it in train:
                train.remove(it)
                holdout.append(it)

    TRAIN.parent.mkdir(parents=True, exist_ok=True)
    TRAIN.write_text(json.dumps(train, indent=2, ensure_ascii=False))
    HOLDOUT.write_text(json.dumps(holdout, indent=2, ensure_ascii=False))

    overlap_q = {i["question"].lower() for i in train} & {i["question"].lower() for i in holdout}
    overlap_pair = {
        (i["question"].lower(), i["gold_chunk_id"]) for i in train
    } & {
        (i["question"].lower(), i["gold_chunk_id"]) for i in holdout
    }
    print(f"train: {len(train)}, holdout: {len(holdout)}")
    print(f"question overlap: {len(overlap_q)}")
    print(f"pair overlap: {len(overlap_pair)}")
    if overlap_q or overlap_pair:
        raise SystemExit("leakage detected")


if __name__ == "__main__":
    main()
