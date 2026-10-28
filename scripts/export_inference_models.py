import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer


class EmbeddingWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).last_hidden_state


class RerankerWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return outputs.logits.squeeze(-1)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(output_dir: Path, entries: dict) -> None:
    (output_dir / "manifest.json").write_text(json.dumps(entries, indent=2, sort_keys=True))


def _export_common(wrapper, tokenizer, dummy, output_dir: Path, output_name: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "model.onnx"
    torch.onnx.export(
        wrapper,
        dummy,
        str(onnx_path),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=[output_name],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "token_type_ids": {0: "batch", 1: "seq"},
            output_name: {0: "batch", 1: "seq"} if output_name == "last_hidden_state" else {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
        do_constant_folding=True,
    )
    tokenizer.save_pretrained(str(output_dir))
    # validate
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    # pytorch reference
    with torch.no_grad():
        pt_out = wrapper(*dummy).cpu().numpy()
    ort_out = sess.run(None, {
        "input_ids": dummy[0].cpu().numpy().astype("int64"),
        "attention_mask": dummy[1].cpu().numpy().astype("int64"),
        "token_type_ids": dummy[2].cpu().numpy().astype("int64"),
    })[0]
    assert np.allclose(pt_out, ort_out, atol=1e-4), "ONNX parity check failed"
    return onnx_path


def export_embedding(model_name: str, revision: str, output_dir: Path) -> dict:
    model = AutoModel.from_pretrained(model_name, revision=revision)
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    wrapper = EmbeddingWrapper(model).eval()
    dummy = (
        torch.randint(0, 100, (1, 16)),
        torch.ones(1, 16, dtype=torch.long),
        torch.zeros(1, 16, dtype=torch.long),
    )
    onnx_path = _export_common(wrapper, tokenizer, dummy, output_dir, "last_hidden_state")
    # fix: pass wrapper correctly (already)
    entry = {
        "model": model_name,
        "revision": revision,
        "output": "last_hidden_state",
        "sha256": sha256_file(onnx_path),
    }
    return entry


def export_reranker(model_name: str, revision: str, output_dir: Path) -> dict:
    model = AutoModelForSequenceClassification.from_pretrained(model_name, revision=revision)
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    wrapper = RerankerWrapper(model).eval()
    dummy = (
        torch.randint(0, 100, (1, 16)),
        torch.ones(1, 16, dtype=torch.long),
        torch.zeros(1, 16, dtype=torch.long),
    )
    onnx_path = _export_common(wrapper, tokenizer, dummy, output_dir, "logits")
    entry = {
        "model": model_name,
        "revision": revision,
        "output": "logits",
        "sha256": sha256_file(onnx_path),
    }
    return entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="models/inference")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding-revision", default="main")
    parser.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--reranker-revision", default="main")
    args = parser.parse_args()

    base = Path(args.output_dir)
    emb_dir = base / "embedding"
    rer_dir = base / "reranker"

    emb_entry = export_embedding(args.embedding_model, args.embedding_revision, emb_dir)
    rer_entry = export_reranker(args.reranker_model, args.reranker_revision, rer_dir)

    write_manifest(base, {"embedding": emb_entry, "reranker": rer_entry})
    print(f"Exported to {base}")


if __name__ == "__main__":
    main()
