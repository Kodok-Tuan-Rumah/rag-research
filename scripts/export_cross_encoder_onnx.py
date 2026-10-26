"""Export sentence-transformers cross-encoder to ONNX format with regression head.

Uses torch.jit.trace for compatibility.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch
from sentence_transformers import CrossEncoder

logger = logging.getLogger("export_cross_encoder_onnx")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    output_dir = Path("models")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading {model_name} ...")
    model = CrossEncoder(model_name)
    model.eval()

    # The CrossEncoder wraps BertForSequenceClassification
    bert_model = model.model  # BertForSequenceClassification
    classifier = bert_model.classifier  # Linear layer

    # Create dummy inputs
    dummy_query = "What is self-attention?"
    dummy_text = "Self-attention allows the model to attend to different positions."

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(
        dummy_query,
        dummy_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]

    logger.info(f"Input shapes: {input_ids.shape}")

    class CrossEncoderWrapper(torch.nn.Module):
        def __init__(self, bert_model, classifier):
            super().__init__()
            self.bert_model = bert_model
            self.classifier = classifier

        def forward(self, input_ids, attention_mask, token_type_ids):
            outputs = self.bert_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            # BertForSequenceClassification returns (loss, logits) or just logits
            # But we want to do our own classifier application for ONNX compatibility
            # Get base model output
            base_outputs = self.bert_model.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            last_hidden_state = base_outputs[0]  # [batch_size, seq_len, hidden_dim]
            cls_embedding = last_hidden_state[:, 0, :]  # [batch_size, hidden_dim]
            logits = self.classifier(cls_embedding)  # [batch_size, 1]
            return logits.squeeze(-1)  # [batch_size]

    wrapper = CrossEncoderWrapper(bert_model, classifier)
    wrapper.eval()

    # Verify the wrapper works
    with torch.no_grad():
        test_output = wrapper(input_ids, attention_mask, token_type_ids)
    logger.info(f"Test output shape: {test_output.shape}, value: {test_output.item():.4f}")

    # Compare with model.predict
    with torch.no_grad():
        pt_scores = model.predict([(dummy_query, dummy_text)])
    logger.info(f"CrossEncoder.predict: {pt_scores[0]:.4f}")

    logger.info("Exporting to ONNX with torch.onnx.export ...")
    torch.onnx.export(
        wrapper,
        (input_ids, attention_mask, token_type_ids),
        str(output_dir / "model.onnx"),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "token_type_ids": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        },
        dynamo=False,
    )

    # Save tokenizer
    tokenizer.save_pretrained(output_dir)

    logger.info(f"Model saved to {output_dir / 'model.onnx'}")
    logger.info(f"Tokenizer saved to {output_dir / 'tokenizer.json'}")

    # Verify with onnxruntime
    import onnxruntime as ort
    import numpy as np

    sess = ort.InferenceSession(str(output_dir / "model.onnx"))
    print(f"\nVerification:")
    print(f"  Input names: {[i.name for i in sess.get_inputs()]}")
    print(f"  Output names: {[o.name for o in sess.get_outputs()]}")
    print(f"  Output shapes: {[o.shape for o in sess.get_outputs()]}")

    ort_inputs = {
        "input_ids": input_ids.numpy().astype(np.int64),
        "attention_mask": attention_mask.numpy().astype(np.int64),
        "token_type_ids": token_type_ids.numpy().astype(np.int64),
    }
    ort_outputs = sess.run(None, ort_inputs)
    print(f"  ONNX output: {ort_outputs[0]}")

    print(f"  PyTorch output: {test_output.item():.6f}")
    print(f"  Match: {np.allclose(ort_outputs[0], test_output.numpy(), atol=1e-4)}")


if __name__ == "__main__":
    main()
