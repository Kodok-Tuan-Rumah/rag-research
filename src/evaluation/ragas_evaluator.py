from __future__ import annotations

import json
import os
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")

# LangChain LLM + Embeddings
from langchain_groq import ChatGroq
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AnswerCorrectness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)
from ragas.run_config import RunConfig

import src._compat  # noqa: F401  must precede ragas; neutralizes nest_asyncio on 3.14

try:
    from langchain_google_genai import ChatGoogleGenerativeAI

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
from langchain_huggingface import HuggingFaceEmbeddings

from src.api.config import settings

# Config
DEFAULT_LLM_MODEL = settings.groq_model_fast
DEFAULT_EMBED_MODEL = settings.embed_model


@dataclass
class RAGSample:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str | None = None


@dataclass
class EvalResult:
    question: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float | None
    context_recall: float | None
    answer_correctness: float | None
    latency_ms: float
    metadata: dict = field(default_factory=dict)


@dataclass
class AblationResult:
    exp_id: str
    config: dict
    samples: list[EvalResult]
    avg_faithfulness: float
    avg_answer_relevancy: float
    avg_context_precision: float
    avg_context_recall: float
    avg_answer_correctness: float | None
    avg_latency_ms: float
    hallucination_rate: float
    total_samples: int
    nan_counts: dict = field(default_factory=dict)


# RAGAS Evaluator
class RAGASEvaluator:
    def __init__(
        self,
        groq_api_key=None,
        llm_model=DEFAULT_LLM_MODEL,
        embed_model=DEFAULT_EMBED_MODEL,
        include_answer_correctness=False,
        llm_provider="groq",
    ):
        print(f"[RAGASEvaluator] Init Evaluation LLM: {llm_model} (provider: {llm_provider})")

        if llm_provider == "gemini" and GEMINI_AVAILABLE:
            gemini_key = os.getenv("GOOGLE_API_KEY")
            if not gemini_key:
                raise ValueError("GOOGLE_API_KEY tidak ditemukan.")
            self.llm = ChatGoogleGenerativeAI(
                model=llm_model,
                google_api_key=gemini_key,
                temperature=0,
            )
        else:
            api_key = groq_api_key or os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY tidak ditemukan.")
            self.llm = ChatGroq(
                groq_api_key=api_key,
                model_name=llm_model,
                temperature=0,
                max_retries=10,
                model_kwargs={"response_format": {"type": "json_object"}},
            )

        print(f"[RAGASEvaluator] Init Embeddings: {embed_model}")
        self.embeddings = HuggingFaceEmbeddings(model_name=embed_model)
        self.include_answer_correctness = include_answer_correctness

        # CONFIG SUPER SABAR
        self.run_config = RunConfig(max_retries=20, timeout=180, max_workers=1)
        self.ragas_llm = LangchainLLMWrapper(self.llm, run_config=self.run_config)
        self.ragas_embeddings = LangchainEmbeddingsWrapper(self.embeddings)

        # Bind metrics
        self._faithfulness = Faithfulness(llm=self.ragas_llm)
        self._answer_relevancy = AnswerRelevancy(
            llm=self.ragas_llm, embeddings=self.ragas_embeddings
        )
        self._context_precision = ContextPrecision(llm=self.ragas_llm)
        self._context_recall = ContextRecall(llm=self.ragas_llm)
        self._answer_correctness = AnswerCorrectness(llm=self.ragas_llm)

        self.metrics_no_ref = [self._faithfulness, self._answer_relevancy]
        self.metrics_with_ref = [
            self._faithfulness,
            self._answer_relevancy,
            self._context_precision,
            self._context_recall,
        ]
        if include_answer_correctness:
            self.metrics_with_ref.append(self._answer_correctness)

    def evaluate(
        self,
        samples: list[RAGSample],
        exp_id: str = "exp_001",
        config: dict | None = None,
    ) -> AblationResult:
        print(f"\n[RAGASEvaluator] Evaluating {len(samples)} samples for {exp_id}...")
        t0 = time.time()
        has_reference = all(s.ground_truth for s in samples)
        active_metrics = self.metrics_with_ref if has_reference else self.metrics_no_ref

        eval_dataset = EvaluationDataset(
            samples=[
                SingleTurnSample(
                    user_input=s.question,
                    reference=s.ground_truth,
                    response=s.answer,
                    retrieved_contexts=s.contexts,
                )
                for s in samples
            ]
        )

        ragas_result = evaluate(
            dataset=eval_dataset,
            metrics=active_metrics,
            llm=self.llm,
            embeddings=self.embeddings,
            run_config=self.run_config,
        )

        ragas_df = ragas_result.to_pandas()
        elapsed_ms = (time.time() - t0) * 1000
        per_sample_ms = elapsed_ms / len(samples)

        eval_results = []
        for i, sample in enumerate(samples):
            row = ragas_df.iloc[i]
            eval_results.append(
                EvalResult(
                    question=sample.question,
                    faithfulness=float(row.get("faithfulness", 0.0)),
                    answer_relevancy=float(row.get("answer_relevancy", 0.0)),
                    context_precision=float(row.get("context_precision", 0.0))
                    if "context_precision" in row
                    else None,
                    context_recall=float(row.get("context_recall", 0.0))
                    if "context_recall" in row
                    else None,
                    answer_correctness=float(row.get("answer_correctness", 0.0))
                    if "answer_correctness" in row
                    else None,
                    latency_ms=per_sample_ms,
                    metadata={"exp_id": exp_id},
                )
            )

        def avg(key):
            vals = [
                getattr(r, key)
                for r in eval_results
                if getattr(r, key) is not None and not np.isnan(getattr(r, key))
            ]
            if not vals:
                return 0.0
            return sum(vals) / len(vals)

        def nan_count(key):
            return sum(
                1 for r in eval_results if getattr(r, key) is None or np.isnan(getattr(r, key))
            )

        nan_counts = {
            key: nan_count(key)
            for key in (
                "faithfulness",
                "answer_relevancy",
                "context_precision",
                "context_recall",
                "answer_correctness",
            )
        }
        empty = [
            key for key, count in nan_counts.items() if count == len(eval_results) and eval_results
        ]
        if empty:
            raise RuntimeError(
                "RAGAS produced no usable score for: "
                + ", ".join(empty)
                + ". Every scoring job failed. Scores would be reported as "
                "0.0000, which is not a measurement. Re-run with "
                "raise_exceptions=True to see the underlying error."
            )

        successful_faithfulness = [
            r.faithfulness for r in eval_results if not np.isnan(r.faithfulness)
        ]

        if successful_faithfulness:
            hallucination_count = sum(1 for v in successful_faithfulness if v < 0.5)
            hallucination_rate = hallucination_count / len(successful_faithfulness)
        else:
            hallucination_rate = 0.0

        result = AblationResult(
            exp_id=exp_id,
            config=config or {},
            samples=eval_results,
            avg_faithfulness=avg("faithfulness"),
            avg_answer_relevancy=avg("answer_relevancy"),
            avg_context_precision=avg("context_precision"),
            avg_context_recall=avg("context_recall"),
            avg_answer_correctness=avg("answer_correctness")
            if self.include_answer_correctness
            else None,
            avg_latency_ms=avg("latency_ms"),
            hallucination_rate=hallucination_rate,
            total_samples=len(samples),
            nan_counts=nan_counts,
        )

        self._print_summary(result)
        return result

    def _print_summary(self, result: AblationResult) -> None:
        print(f"\n{'=' * 55}")
        print(f"  [OK] RAGAS Results — {result.exp_id}")
        print(f"{'=' * 55}")
        print(f"  Faithfulness       : {result.avg_faithfulness:.4f}")
        print(f"  Answer Relevancy   : {result.avg_answer_relevancy:.4f}")
        print(f"  Context Precision  : {result.avg_context_precision:.4f}")
        print(f"  Context Recall     : {result.avg_context_recall:.4f}")
        if result.avg_answer_correctness is not None:
            print(f"  Answer Correctness : {result.avg_answer_correctness:.4f}")
        print(f"  Hallucination Rate : {result.hallucination_rate:.1%}")
        print(f"  Samples            : {result.total_samples}")
        if result.nan_counts:
            degraded = {k: v for k, v in result.nan_counts.items() if v}
            if degraded:
                detail = ", ".join(f"{k}={v}/{result.total_samples}" for k, v in degraded.items())
                print(f"  [WARNING] incomplete metrics: {detail}")
        print(f"{'=' * 55}\n")

    def save_result(self, result: AblationResult, output_path: str) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        existing = []
        if Path(output_path).exists():
            with open(output_path) as f:
                existing = json.load(f)
        existing.append(asdict(result))
        with open(output_path, "w") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
        print(f"[Saved] Results → {output_path}")


# CLI Smoke Test
if __name__ == "__main__":
    samples = [
        RAGSample(
            question="What is RAG?",
            answer="RAG is Retrieval-Augmented Generation.",
            contexts=["RAG combines retrieval with generation."],
            ground_truth="RAG stands for Retrieval-Augmented Generation.",
        )
    ]
    if os.getenv("GROQ_API_KEY"):
        evaluator = RAGASEvaluator(include_answer_correctness=True)
        evaluator.evaluate(samples, exp_id="smoke_test")
    else:
        print("Set GROQ_API_KEY dulu!")
