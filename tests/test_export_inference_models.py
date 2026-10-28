import hashlib
import json
from pathlib import Path

import torch
from transformers import BertConfig, BertForSequenceClassification, BertModel

from scripts.export_inference_models import (
    EmbeddingWrapper,
    RerankerWrapper,
    sha256_file,
    write_manifest,
)


def test_sha256_file(tmp_path: Path):
    path = tmp_path / "payload.bin"
    path.write_bytes(b"abc")
    assert sha256_file(path) == hashlib.sha256(b"abc").hexdigest()


def test_write_manifest(tmp_path: Path):
    write_manifest(tmp_path, {"embedding": {"revision": "abc", "sha256": "def"}})
    payload = json.loads((tmp_path / "manifest.json").read_text())
    assert payload["embedding"]["revision"] == "abc"


def test_embedding_wrapper_output_shape():
    config = BertConfig(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        vocab_size=100,
    )
    wrapper = EmbeddingWrapper(BertModel(config)).eval()
    input_ids = torch.randint(0, 100, (2, 5))
    attention_mask = torch.ones(2, 5, dtype=torch.long)
    token_type_ids = torch.zeros(2, 5, dtype=torch.long)
    with torch.no_grad():
        output = wrapper(input_ids, attention_mask, token_type_ids)
    assert output.shape == (2, 5, 32)


def test_reranker_wrapper_output_shape():
    config = BertConfig(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        vocab_size=100,
        num_labels=1,
    )
    wrapper = RerankerWrapper(BertForSequenceClassification(config)).eval()
    input_ids = torch.randint(0, 100, (2, 5))
    attention_mask = torch.ones(2, 5, dtype=torch.long)
    token_type_ids = torch.zeros(2, 5, dtype=torch.long)
    with torch.no_grad():
        output = wrapper(input_ids, attention_mask, token_type_ids)
    assert output.shape == (2,)
