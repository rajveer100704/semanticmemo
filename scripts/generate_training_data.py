#!/usr/bin/env python3
"""Generate equivalence-classifier training data with a local Ollama model.

Reads the hand-authored corpus under ``data/corpus/`` and produces labeled
prompt pairs in the JSONL shape consumed by ``smartmemo train-classifier``:

* **positives**  -- LLM paraphrases that preserve intent (label 1)
* **hard negatives** -- same object, opposite action; built from templates so
  the labels are correct by construction (label 0)
* **easy negatives** -- unrelated prompts paired at random (label 0)

Only the standard library is used; the Ollama HTTP API is called directly.
Paraphrase responses are cached on disk, so reruns are cheap and the committed
dataset can be regenerated deterministically. Train and validation splits never
share a base prompt, so paraphrase text cannot leak across the split.

Usage::

    python scripts/generate_training_data.py
    python scripts/generate_training_data.py --max-base 4 --max-templates 4  # smoke
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = REPO_ROOT / "data" / "corpus"
DEFAULT_OUT = REPO_ROOT / "data" / "training"

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")

PARAPHRASE_SYSTEM = (
    "You rewrite task instructions while preserving their exact meaning and the "
    "specific action requested. You never change what is being asked, and you "
    "never answer the instruction."
)


def ollama_chat(
    *,
    host: str,
    model: str,
    system: str,
    user: str,
    temperature: float,
    seed: int,
    cache: dict[str, str],
    timeout: int = 180,
) -> str:
    """Call the Ollama chat endpoint with on-disk caching and retries."""

    cache_key = hashlib.sha256(
        json.dumps([model, system, user, temperature, seed], sort_keys=True).encode()
    ).hexdigest()
    if cache_key in cache:
        return cache[cache_key]

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "think": False,  # qwen3 is a thinking model; keep output clean
        "options": {"temperature": temperature, "seed": seed},
    }
    request = urllib.request.Request(  # noqa: S310 - localhost Ollama only
        f"{host}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
                body = json.loads(response.read())
            content = str(body.get("message", {}).get("content", ""))
            cache[cache_key] = content
            return content
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            last_error = error
            time.sleep(2 * (attempt + 1))
    msg = f"Ollama request failed after 3 attempts: {last_error}"
    raise RuntimeError(msg)


def paraphrase(
    text: str,
    count: int,
    *,
    host: str,
    model: str,
    seed: int,
    cache: dict[str, str],
) -> list[str]:
    """Return up to ``count`` intent-preserving paraphrases of ``text``."""

    user = (
        f"Rewrite the following instruction in {count} different ways. Preserve "
        "the exact intent and the specific action requested - only vary the "
        "wording, phrasing, and sentence structure. Do not answer the "
        f"instruction. Return exactly {count} rewrites, one per line, with no "
        f"numbering, quotes, or commentary.\n\nInstruction: {text}"
    )
    raw = ollama_chat(
        host=host,
        model=model,
        system=PARAPHRASE_SYSTEM,
        user=user,
        temperature=0.8,
        seed=seed,
        cache=cache,
    )
    raw = THINK_RE.sub("", raw)

    results: list[str] = []
    seen = {text.strip().lower()}
    for line in raw.splitlines():
        cleaned = LIST_PREFIX_RE.sub("", line.strip()).strip().strip("\"'").strip()
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        results.append(cleaned)
    return results[:count]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped:
            rows.append(json.loads(stripped))
    return rows


def stratified_split(
    items: list[dict[str, Any]],
    ratio: float,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split items into (train, validation), stratified by ``domain``."""

    by_domain: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        by_domain.setdefault(str(item["domain"]), []).append(item)

    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for _, group in sorted(by_domain.items()):
        shuffled = group[:]
        rng.shuffle(shuffled)
        cut = max(1, min(len(shuffled) - 1, round(len(shuffled) * ratio)))
        train.extend(shuffled[:cut])
        validation.extend(shuffled[cut:])
    return train, validation


def build_concepts(
    base_prompts: list[dict[str, Any]],
    templates: list[dict[str, Any]],
    *,
    n_para_base: int,
    n_para_template: int,
    host: str,
    model: str,
    seed: int,
    cache: dict[str, str],
) -> list[dict[str, Any]]:
    """Materialize equivalence concepts: each holds intent-equivalent renderings."""

    concepts: list[dict[str, Any]] = []
    total = len(base_prompts) + len(templates)
    done = 0

    for index, prompt in enumerate(base_prompts):
        text = str(prompt["text"])
        renders = [
            text,
            *paraphrase(text, n_para_base, host=host, model=model, seed=seed + index, cache=cache),
        ]
        concepts.append({"domain": str(prompt["domain"]), "renders": renders, "template": None})
        done += 1
        print(f"  paraphrased {done}/{total}", file=sys.stderr)

    for index, template in enumerate(templates):
        obj = str(template["object"])
        render_a = f"{template['action_a']} {obj}."
        render_b = f"{template['action_b']} {obj}."
        renders_a = [
            render_a,
            *paraphrase(
                render_a,
                n_para_template,
                host=host,
                model=model,
                seed=seed + 10_000 + index,
                cache=cache,
            ),
        ]
        renders_b = [
            render_b,
            *paraphrase(
                render_b,
                n_para_template,
                host=host,
                model=model,
                seed=seed + 20_000 + index,
                cache=cache,
            ),
        ]
        domain = str(template["domain"])
        concepts.append({"domain": domain, "renders": renders_a, "template": index, "side": "a"})
        concepts.append({"domain": domain, "renders": renders_b, "template": index, "side": "b"})
        done += 1
        print(f"  paraphrased {done}/{total}", file=sys.stderr)

    return concepts


def build_pairs(
    concepts: list[dict[str, Any]],
    *,
    split: str,
    rng: random.Random,
    hard_neg_per_template: int,
    neg_ratio: float,
) -> list[dict[str, Any]]:
    """Turn concepts into labeled prompt pairs for one dataset split."""

    positives: list[dict[str, Any]] = []
    for concept in concepts:
        for prompt_a, prompt_b in itertools.combinations(concept["renders"], 2):
            positives.append(_pair(prompt_a, prompt_b, 1, concept["domain"], "paraphrase", split))

    # Hard negatives: opposite sides of the same action template.
    by_template: dict[int, dict[str, dict[str, Any]]] = {}
    for concept in concepts:
        if concept["template"] is not None:
            by_template.setdefault(concept["template"], {})[concept["side"]] = concept

    hard_negatives: list[dict[str, Any]] = []
    for sides in by_template.values():
        if "a" not in sides or "b" not in sides:
            continue
        side_a, side_b = sides["a"], sides["b"]
        cross = [(x, y) for x in side_a["renders"] for y in side_b["renders"]]
        rng.shuffle(cross)
        for prompt_a, prompt_b in cross[:hard_neg_per_template]:
            hard_negatives.append(
                _pair(prompt_a, prompt_b, 0, side_a["domain"], "opposite-action", split)
            )

    # Easy/medium negatives: random pairs of unrelated concepts.
    flat = [
        (render, idx, concept["template"], concept["domain"])
        for idx, concept in enumerate(concepts)
        for render in concept["renders"]
    ]
    target = max(0, round(len(positives) * neg_ratio) - len(hard_negatives))
    easy_negatives: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    attempts = 0
    while len(easy_negatives) < target and attempts < max(target * 60, 1):
        attempts += 1
        left = rng.choice(flat)
        right = rng.choice(flat)
        if left[1] == right[1] or left[0] == right[0]:
            continue
        if left[2] is not None and left[2] == right[2]:
            continue  # same template -> already a hard negative
        key = tuple(sorted((left[0], right[0])))
        if key in seen:
            continue
        seen.add(key)
        domain = left[3] if left[3] == right[3] else None
        easy_negatives.append(_pair(left[0], right[0], 0, domain, "unrelated", split))

    # The classifier's pair features are order-sensitive, so emit every pair in
    # both orders. Without this the model overfits argument order -- e.g. it
    # learns "(approve, deny)" is a negative but scores "(deny, approve)" as a hit.
    pairs: list[dict[str, Any]] = []
    for pair in positives + hard_negatives + easy_negatives:
        pairs.append(pair)
        pairs.append(
            _pair(
                pair["prompt_b"],
                pair["prompt_a"],
                pair["label"],
                pair["domain"],
                pair["source"],
                split,
            )
        )
    rng.shuffle(pairs)
    return pairs


def _pair(
    prompt_a: str,
    prompt_b: str,
    label: int,
    domain: str | None,
    source: str,
    split: str,
) -> dict[str, Any]:
    return {
        "prompt_a": prompt_a,
        "prompt_b": prompt_b,
        "label": label,
        "domain": domain,
        "source": source,
        "split": split,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, sort_keys=True) for row in rows]
    path.write_text("\n".join(lines) + "\n")


def summarize(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "total": len(rows),
        "positives": sum(1 for row in rows if row["label"] == 1),
        "negatives": sum(1 for row in rows if row["label"] == 0),
        "opposite_action": sum(1 for row in rows if row["source"] == "opposite-action"),
        "unrelated": sum(1 for row in rows if row["source"] == "unrelated"),
        "paraphrase": sum(1 for row in rows if row["source"] == "paraphrase"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-para-base", type=int, default=6)
    parser.add_argument("--n-para-template", type=int, default=4)
    parser.add_argument("--hard-neg-per-template", type=int, default=10)
    parser.add_argument("--neg-ratio", type=float, default=1.0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--max-base", type=int, default=None, help="smoke-test cap")
    parser.add_argument("--max-templates", type=int, default=None, help="smoke-test cap")
    args = parser.parse_args()

    base_prompts = read_jsonl(args.corpus_dir / "base_prompts.jsonl")
    templates = read_jsonl(args.corpus_dir / "action_templates.jsonl")
    if args.max_base is not None:
        base_prompts = base_prompts[: args.max_base]
    if args.max_templates is not None:
        templates = templates[: args.max_templates]

    cache_path = args.out_dir / ".paraphrase_cache.json"
    cache: dict[str, str] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
        print(f"loaded {len(cache)} cached paraphrase responses", file=sys.stderr)

    split_rng = random.Random(args.seed)
    train_base, val_base = stratified_split(base_prompts, args.train_ratio, split_rng)
    train_templates, val_templates = stratified_split(templates, args.train_ratio, split_rng)
    print(
        f"corpus: {len(base_prompts)} base prompts, {len(templates)} templates | "
        f"train base={len(train_base)} val base={len(val_base)} "
        f"train tmpl={len(train_templates)} val tmpl={len(val_templates)}",
        file=sys.stderr,
    )

    try:
        print("generating train concepts...", file=sys.stderr)
        train_concepts = build_concepts(
            train_base,
            train_templates,
            n_para_base=args.n_para_base,
            n_para_template=args.n_para_template,
            host=args.ollama_host,
            model=args.model,
            seed=args.seed,
            cache=cache,
        )
        print("generating validation concepts...", file=sys.stderr)
        val_concepts = build_concepts(
            val_base,
            val_templates,
            n_para_base=args.n_para_base,
            n_para_template=args.n_para_template,
            host=args.ollama_host,
            model=args.model,
            seed=args.seed + 500_000,
            cache=cache,
        )
    finally:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, sort_keys=True))
        print(f"saved {len(cache)} cached responses to {cache_path}", file=sys.stderr)

    pair_rng = random.Random(args.seed)
    train_rows = build_pairs(
        train_concepts,
        split="train",
        rng=pair_rng,
        hard_neg_per_template=args.hard_neg_per_template,
        neg_ratio=args.neg_ratio,
    )
    val_rows = build_pairs(
        val_concepts,
        split="validation",
        rng=pair_rng,
        hard_neg_per_template=args.hard_neg_per_template,
        neg_ratio=args.neg_ratio,
    )

    train_path = args.out_dir / "pairs_v2.train.jsonl"
    val_path = args.out_dir / "pairs_v2.validation.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model": args.model,
        "ollama_host": args.ollama_host,
        "seed": args.seed,
        "n_para_base": args.n_para_base,
        "n_para_template": args.n_para_template,
        "hard_neg_per_template": args.hard_neg_per_template,
        "neg_ratio": args.neg_ratio,
        "train_ratio": args.train_ratio,
        "corpus": {
            "base_prompts": len(base_prompts),
            "action_templates": len(templates),
        },
        "train": summarize(train_rows),
        "validation": summarize(val_rows),
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"\nwrote {train_path}  -> {summarize(train_rows)}")
    print(f"wrote {val_path}  -> {summarize(val_rows)}")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
