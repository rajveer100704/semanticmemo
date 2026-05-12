"""Train a tiny classifier checkpoint from the seed customer-support fixture.

This example uses HashEmbeddingProvider so it runs without model downloads. Real
training should use SentenceTransformerEmbeddingProvider via `equivcache[ml]`.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from equivcache.classifier import TrainingConfig, load_pair_records, train_classifier
from equivcache.embedding import HashEmbeddingProvider


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    dataset_path = project_root / "data" / "fixtures" / "customer_support_pairs.jsonl"
    train_records = load_pair_records(dataset_path, split="train", domain="customer-support")
    validation_records = load_pair_records(
        dataset_path,
        split="validation",
        domain="customer-support",
    )
    provider = HashEmbeddingProvider(dim=64)

    with TemporaryDirectory() as tmpdir:
        result = train_classifier(
            train_records=train_records,
            validation_records=validation_records,
            embedding_provider=provider,
            output_path=Path(tmpdir) / "classifier-smoke.pt",
            config=TrainingConfig(epochs=2, batch_size=4, threshold=0.85),
        )
        print(f"checkpoint={result.checkpoint_path}")
        print(f"train_examples={result.train_examples}")
        print(f"validation_examples={result.validation_examples}")
        print(f"final_train_loss={result.final_train_loss:.3f}")


if __name__ == "__main__":
    main()
