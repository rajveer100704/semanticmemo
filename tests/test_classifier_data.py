from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from semanticmemo.classifier.data import EmbeddedPairDataset, PairRecord, load_pair_records
from semanticmemo.types import FloatVector


class Provider:
    dim = 2

    def embed(self, text: str) -> FloatVector:
        if "refund" in text:
            return np.array([1, 0], dtype=np.float32)
        return np.array([0, 1], dtype=np.float32)


def test_load_pair_records_filters_split_and_domain(tmp_path: Path) -> None:
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"prompt_a":"refund a","prompt_b":"refund b","label":1,'
                '"domain":"customer-support","split":"train"}',
                '{"prompt_a":"legal a","prompt_b":"legal b","label":0,'
                '"domain":"legal","split":"train"}',
                '{"prompt_a":"refund c","prompt_b":"refund d","label":1,'
                '"domain":"customer-support","split":"test"}',
            ]
        )
    )

    records = load_pair_records(path, split="train", domain="customer-support")

    assert len(records) == 1
    assert records[0].label == 1
    assert records[0].split == "train"


def test_pair_record_rejects_invalid_label() -> None:
    with pytest.raises(ValueError, match="label must be 0 or 1"):
        PairRecord.from_mapping({"prompt_a": "a", "prompt_b": "b", "label": 2})


def test_embedded_pair_dataset_embeds_records() -> None:
    records = [
        PairRecord(prompt_a="refund one", prompt_b="refund two", label=1),
        PairRecord(prompt_a="refund one", prompt_b="password reset", label=0),
    ]

    dataset = EmbeddedPairDataset(records, Provider())
    emb_a, emb_b, label = dataset[0]

    assert len(dataset) == 2
    assert emb_a.shape == (2,)
    assert emb_b.shape == (2,)
    assert float(label) == 1.0
