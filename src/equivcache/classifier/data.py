"""Dataset helpers for labeled prompt-pair training data."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from equivcache.embedding.service import normalize
from equivcache.types import EmbeddingProvider

_torch: Any = torch


@dataclass(frozen=True)
class PairRecord:
    """One labeled prompt pair for equivalence training or evaluation."""

    prompt_a: str
    prompt_b: str
    label: int
    domain: str | None = None
    source: str | None = None
    split: str = "train"

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> PairRecord:
        missing = {"prompt_a", "prompt_b", "label"} - value.keys()
        if missing:
            msg = f"Pair record is missing required fields: {sorted(missing)}"
            raise ValueError(msg)
        prompt_a = str(value["prompt_a"])
        prompt_b = str(value["prompt_b"])
        label = int(value["label"])
        if not prompt_a or not prompt_b:
            msg = "prompt_a and prompt_b must be non-empty"
            raise ValueError(msg)
        if label not in {0, 1}:
            msg = f"label must be 0 or 1, got {label}"
            raise ValueError(msg)
        return cls(
            prompt_a=prompt_a,
            prompt_b=prompt_b,
            label=label,
            domain=str(value["domain"]) if value.get("domain") is not None else None,
            source=str(value["source"]) if value.get("source") is not None else None,
            split=str(value.get("split", "train")),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "prompt_a": self.prompt_a,
            "prompt_b": self.prompt_b,
            "label": self.label,
            "domain": self.domain,
            "source": self.source,
            "split": self.split,
        }


def load_pair_records(
    path: Path | str,
    *,
    split: str | None = None,
    domain: str | None = None,
) -> list[PairRecord]:
    """Load JSONL pair records, optionally filtered by split and domain."""

    records: list[PairRecord] = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = PairRecord.from_mapping(json.loads(stripped))
        except json.JSONDecodeError as exc:
            msg = f"Invalid JSON on line {line_number} of {path}"
            raise ValueError(msg) from exc
        if split is not None and record.split != split:
            continue
        if domain is not None and record.domain != domain:
            continue
        records.append(record)
    return records


def write_pair_records(path: Path | str, records: Iterable[PairRecord]) -> None:
    """Write pair records as JSONL."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record.to_mapping(), sort_keys=True) for record in records]
    output_path.write_text("\n".join(lines) + "\n")


class EmbeddedPairDataset(Dataset):
    """In-memory tensor dataset produced from prompt pairs and an embedding provider."""

    def __init__(
        self,
        records: Sequence[PairRecord],
        embedding_provider: EmbeddingProvider,
    ) -> None:
        if not records:
            msg = "EmbeddedPairDataset requires at least one record"
            raise ValueError(msg)
        self.records = list(records)
        self.embeddings_a = [
            _torch.tensor(
                normalize(embedding_provider.embed(record.prompt_a)),
                dtype=_torch.float32,
            )
            for record in records
        ]
        self.embeddings_b = [
            _torch.tensor(
                normalize(embedding_provider.embed(record.prompt_b)),
                dtype=_torch.float32,
            )
            for record in records
        ]
        self.labels = [
            _torch.tensor(float(record.label), dtype=_torch.float32) for record in records
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.embeddings_a[index], self.embeddings_b[index], self.labels[index]
