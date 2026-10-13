"""Exploratory data analysis of CoNLL-2003: sizes, tag/entity distributions, lengths, OOV rates.

    python -m ner.eda
"""
from __future__ import annotations

import json
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import ENTITY_COLORS, ENTITY_TYPES, RESULTS_DIR
from .data import get_spans, invalid_transitions, load_conll, normalize_word
from .embeddings import load_glove


def main():
    data = load_conll()
    out = RESULTS_DIR / "eda"
    out.mkdir(parents=True, exist_ok=True)

    train_vocab = {w for toks, _ in data["train"] for w in toks}
    all_norm = {normalize_word(w) for s in data.values() for toks, _ in s for w in toks}
    glove = set(load_glove(all_norm))

    stats = {}
    for split, sents in data.items():
        tokens = [w for toks, _ in sents for w in toks]
        tags = Counter(t for _, tg in sents for t in tg)
        ents = Counter(e for _, tg in sents for e, _, _ in get_spans(tg))
        span_lens = Counter(e - s for _, tg in sents for _, s, e in get_spans(tg))
        stats[split] = {
            "sentences": len(sents),
            "tokens": len(tokens),
            "entities": dict(ents),
            "tags": dict(tags),
            "entity_length_distribution": dict(sorted(span_lens.items())),
            "avg_sentence_length": round(len(tokens) / len(sents), 2),
            "max_sentence_length": max(len(t) for t, _ in sents),
            "iob2_violations": sum(invalid_transitions(tg) for _, tg in sents),
            "oov_rate_vs_train_vocab": round(sum(w not in train_vocab for w in tokens) / len(tokens), 4),
            "oov_rate_vs_glove": round(sum(normalize_word(w) not in glove for w in tokens) / len(tokens), 4),
        }
    with open(out / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps({s: {k: v for k, v in d.items() if k != "tags"} for s, d in stats.items()}, indent=2))

    # Entity counts per split
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.27
    for i, split in enumerate(data):
        counts = [stats[split]["entities"].get(t, 0) for t in ENTITY_TYPES]
        ax.bar([x + i * width for x in range(4)], counts, width, label=split)
    ax.set_xticks([x + width for x in range(4)], ENTITY_TYPES)
    ax.set_ylabel("# entity spans")
    ax.set_title("Entity spans per type")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "entity_counts.png", dpi=120)

    # Tag distribution (train)
    fig, ax = plt.subplots(figsize=(8, 4))
    tags = stats["train"]["tags"]
    order = sorted(tags, key=lambda t: (t != "O", t[2:], t))
    colors = ["#bbbbbb" if t == "O" else ENTITY_COLORS[t[2:]] for t in order]
    ax.bar(order, [tags[t] for t in order], color=colors)
    ax.set_yscale("log")
    ax.set_title("IOB2 tag distribution (train, log scale)")
    fig.tight_layout()
    fig.savefig(out / "tag_distribution.png", dpi=120)

    # Sentence lengths
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist([len(t) for t, _ in data["train"]], bins=60, color="#7fb3ff")
    ax.set_xlabel("tokens per sentence")
    ax.set_ylabel("# sentences")
    ax.set_title("Sentence length distribution (train)")
    fig.tight_layout()
    fig.savefig(out / "sentence_lengths.png", dpi=120)
    print(f"Saved figures to {out}")


if __name__ == "__main__":
    main()
