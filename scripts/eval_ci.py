from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RAGASEvaluator = None


@dataclass
class RAGSample:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str

THRESHOLDS = {
    "faithfulness": 0.88,
    "answer_relevancy": 0.88,
    "context_recall": 0.90,
    "hallucination_rate": 0.02,
}

DEFAULT_ADVERSARIAL_DIR = Path("data/adversarial")


def load_json_file(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_samples(path: Path) -> list[RAGSample]:
    if not path.exists():
        raise FileNotFoundError(f"Eval path not found: {path}")

    samples: list[RAGSample] = []
    if path.is_file():
        items = load_json_file(path)
        if isinstance(items, dict):
            items = [items]
    else:
        items = []
        for file_path in sorted(path.glob("*.json")):
            payload = load_json_file(file_path)
            if isinstance(payload, dict):
                items.extend(payload.get("samples", [payload]))
            elif isinstance(payload, list):
                items.extend(payload)

    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"Evaluation row {index} must be a JSON object.")
        question = item.get("question") or item.get("query")
        answer = item.get("answer") or item.get("response")
        contexts = item.get("contexts") or item.get("context") or []
        if isinstance(contexts, str):
            contexts = [contexts]
        if not isinstance(contexts, list) or any(not isinstance(context, str) for context in contexts):
            raise ValueError(f"Evaluation row {index} contexts/context must be a string or list of strings.")
        ground_truth = item.get("ground_truth")
        missing = [
            field for field, value in (
                ("question/query", question),
                ("answer/response", answer),
                ("contexts/context", contexts),
                ("ground_truth", ground_truth),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                f"Evaluation row {index} is not RAGAS-ready; missing/non-empty fields: "
                f"{', '.join(missing)}."
            )
        samples.append(RAGSample(
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth,
        ))

    return samples


def run_ci_eval(eval_dir: Path, output_path: Path) -> int:
    print(f"[CI EVAL] Loading adversarial data from: {eval_dir}")
    samples = load_samples(eval_dir)
    if not samples:
        print(f"No samples found in {eval_dir}")
        return 1

    evaluator_type = RAGASEvaluator
    if evaluator_type is None:
        from src.evaluation.ragas_evaluator import RAGASEvaluator as evaluator_type

    evaluator = evaluator_type(include_answer_correctness=False)
    result = evaluator.evaluate(samples, exp_id="ci_eval")

    def finite_metric(value):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return numeric if math.isfinite(numeric) else None

    summary = {
        "faithfulness": finite_metric(result.avg_faithfulness),
        "answer_relevancy": finite_metric(result.avg_answer_relevancy),
        "context_recall": finite_metric(result.avg_context_recall),
        "hallucination_rate": finite_metric(result.hallucination_rate),
        "total_samples": result.total_samples,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "config": THRESHOLDS}, f, indent=2)

    failures = []
    for metric, threshold in THRESHOLDS.items():
        actual = summary.get(metric)
        if not isinstance(actual, (int, float)) or not math.isfinite(actual):
            failures.append(f"{metric}: metric unavailable or non-finite")
            continue
        if metric == "hallucination_rate":
            if actual > threshold:
                failures.append(f"{metric}: {actual:.3f} > {threshold}")
        else:
            if actual < threshold:
                failures.append(f"{metric}: {actual:.3f} < {threshold}")

    print("\n[CI EVAL] Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    if failures:
        print("\nEVAL GATE FAILED:")
        for failure in failures:
            print(f" - {failure}")
        return 1

    print("\nAll eval gates passed.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG CI evaluation gate")
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=DEFAULT_ADVERSARIAL_DIR,
        help="Path to adversarial evaluation samples",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/ci_eval.json"),
        help="Path to save the CI evaluation summary",
    )
    args = parser.parse_args()

    exit_code = run_ci_eval(args.eval_dir, args.output)
    sys.exit(exit_code)