"""Compare all trained architectures and analyse where the CRF layer helps boundary detection.

    python -m ner.compare
Writes results/comparison.md, results/*.png and results/crf_examples.html.
"""
from __future__ import annotations

import html
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from . import ENTITY_COLORS, ENTITY_TYPES, LABELS, MODELS_DIR, RESULTS_DIR
from .data import get_spans, invalid_transitions, load_conll

NAMES = {"lstm": "LSTM", "bilstm": "BiLSTM", "bilstm_crf": "BiLSTM + CRF", "transformer": "BERT (fine-tuned)"}


def span_errors(gold_tags, pred_tags) -> dict[str, int]:
    """Classify every gold / predicted span of one sentence.

    correct        exact boundaries and type
    type_error     exact boundaries, wrong type
    boundary_error overlaps a gold span of the same type but boundaries differ
    both_error     overlaps a gold span, wrong type and wrong boundaries
    missed         gold span with no overlapping prediction
    spurious       predicted span with no overlapping gold span
    """
    gold, pred = get_spans(gold_tags), get_spans(pred_tags)
    c = dict.fromkeys(["correct", "type_error", "boundary_error", "both_error", "missed", "spurious"], 0)
    overlap = lambda a, b: a[1] < b[2] and b[1] < a[2]
    for p in pred:
        if p in gold:
            c["correct"] += 1
        elif not any(overlap(p, g) for g in gold):
            c["spurious"] += 1
        else:
            g = next(g for g in gold if overlap(p, g))
            same_bounds = (g[1], g[2]) == (p[1], p[2])
            c["type_error" if same_bounds else ("boundary_error" if g[0] == p[0] else "both_error")] += 1
    c["missed"] = sum(not any(overlap(g, p) for p in pred) for g in gold)
    return c


def render_html(tokens, tags) -> str:
    """Tokens with entity spans highlighted in their entity colour."""
    spans = {s: (t, e) for t, s, e in get_spans(tags)}
    out, i = [], 0
    while i < len(tokens):
        if i in spans:
            t, e = spans[i]
            words = html.escape(" ".join(tokens[i:e]))
            out.append(f'<mark style="background:{ENTITY_COLORS[t]};padding:1px 4px;border-radius:4px">{words} '
                       f'<b style="font-size:0.7em">{t}</b></mark>')
            i = e
        else:
            out.append(html.escape(tokens[i]))
            i += 1
    return " ".join(out)


def main():
    data = load_conll()
    gold = [tags for _, tags in data["test"]]
    tokens = [toks for toks, _ in data["test"]]
    models = {}
    for key in NAMES:
        mpath, ppath = RESULTS_DIR / f"{key}_metrics.json", RESULTS_DIR / "predictions" / f"{key}_test.json"
        if mpath.exists() and ppath.exists():
            models[key] = {"metrics": json.loads(mpath.read_text()), "preds": json.loads(ppath.read_text())}
    if not models:
        raise SystemExit("No results found - train some models first.")

    lines = ["# Model comparison (CoNLL-2003 test set, seqeval entity-level metrics)\n"]
    lines.append("| Model | Precision | Recall | F1 | " + " | ".join(f"{t} F1" for t in ENTITY_TYPES) + " | Dev F1 |")
    lines.append("|---" * (5 + len(ENTITY_TYPES)) + "|")
    for key, m in models.items():
        r = m["metrics"]["test"]
        micro = r["micro avg"]
        lines.append(f"| {NAMES[key]} | {micro['precision']:.4f} | {micro['recall']:.4f} | **{micro['f1-score']:.4f}** | "
                     + " | ".join(f"{r[t]['f1-score']:.4f}" for t in ENTITY_TYPES)
                     + f" | {m['metrics']['best_dev_f1']:.4f} |")

    lines.append("\n## Per-entity precision / recall / F1\n")
    for key, m in models.items():
        r = m["metrics"]["test"]
        lines.append(f"**{NAMES[key]}**\n\n| Entity | Precision | Recall | F1 | Support |\n|---|---|---|---|---|")
        for t in ENTITY_TYPES:
            lines.append(f"| {t} | {r[t]['precision']:.4f} | {r[t]['recall']:.4f} | {r[t]['f1-score']:.4f} | {int(r[t]['support'])} |")
        lines.append("")

    # ----- boundary analysis
    lines.append("## Error analysis: boundaries and IOB validity\n")
    lines.append("| Model | Correct | Boundary err. | Type err. | Type+boundary err. | Missed | Spurious | Invalid IOB transitions |")
    lines.append("|---|---|---|---|---|---|---|---|")
    analysis = {}
    for key, m in models.items():
        tot = dict.fromkeys(["correct", "type_error", "boundary_error", "both_error", "missed", "spurious"], 0)
        for g, p in zip(gold, m["preds"]):
            for k, v in span_errors(g, p).items():
                tot[k] += v
        tot["invalid_transitions"] = sum(invalid_transitions(p) for p in m["preds"])
        analysis[key] = tot
        lines.append(f"| {NAMES[key]} | {tot['correct']} | {tot['boundary_error']} | {tot['type_error']} | {tot['both_error']} | "
                     f"{tot['missed']} | {tot['spurious']} | {tot['invalid_transitions']} |")
    with open(RESULTS_DIR / "error_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)

    # ----- examples where the CRF fixes BiLSTM boundaries
    if "bilstm" in models and "bilstm_crf" in models:
        examples = []
        for i, g in enumerate(gold):
            a, b = models["bilstm"]["preds"][i], models["bilstm_crf"]["preds"][i]
            if b == g and a != g and (invalid_transitions(a) or span_errors(g, a)["boundary_error"]) and len(g) <= 30:
                examples.append(i)
        rows = [
            f"<tr><td>{render_html(tokens[i], gold[i])}</td><td>{render_html(tokens[i], models['bilstm']['preds'][i])}</td>"
            f"<td>{render_html(tokens[i], models['bilstm_crf']['preds'][i])}</td></tr>"
            for i in examples[:25]
        ]
        (RESULTS_DIR / "crf_examples.html").write_text(
            "<html><head><meta charset='utf-8'><title>CRF boundary fixes</title><style>body{font-family:sans-serif;margin:24px}"
            "td{border-bottom:1px solid #ddd;padding:8px;vertical-align:top;line-height:1.9}</style></head><body>"
            f"<h2>Test sentences where BiLSTM + CRF is exactly right and the BiLSTM gets a boundary wrong ({len(examples)} total)</h2>"
            "<table><tr><th>Gold</th><th>BiLSTM (softmax)</th><th>BiLSTM + CRF</th></tr>" + "".join(rows) + "</table></body></html>",
            encoding="utf-8",
        )
        lines.append(f"\n{len(examples)} test sentences are tagged perfectly by BiLSTM + CRF while the plain BiLSTM makes a "
                     "boundary error or an invalid IOB transition. Examples are in [crf_examples.html](crf_examples.html).\n")
        for i in examples[:5]:
            fmt = lambda tags: " ".join(f"[{' '.join(tokens[i][s:e])}]{t}" for t, s, e in get_spans(tags))
            lines.append(f"- *{' '.join(tokens[i])}*\n  - BiLSTM: {fmt(models['bilstm']['preds'][i])}\n"
                         f"  - BiLSTM + CRF: {fmt(models['bilstm_crf']['preds'][i])}")
    (RESULTS_DIR / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    # ----- plots
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.8 / len(models)
    cats = ENTITY_TYPES + ["micro avg"]
    for j, (key, m) in enumerate(models.items()):
        vals = [m["metrics"]["test"][c]["f1-score"] for c in cats]
        bars = ax.bar([x + j * width for x in range(len(cats))], vals, width, label=NAMES[key])
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=1)
    ax.set_xticks([x + width * (len(models) - 1) / 2 for x in range(len(cats))], ["PER", "ORG", "LOC", "MISC", "Overall"])
    ax.set_ylim(0.6, 1.0)
    ax.set_ylabel("F1 (test)")
    ax.set_title("Per-entity F1 on CoNLL-2003 test")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "f1_comparison.png", dpi=130)

    fig, ax = plt.subplots(figsize=(9, 4))
    kinds = ["boundary_error", "type_error", "both_error", "missed", "spurious", "invalid_transitions"]
    for j, key in enumerate(analysis):
        bars = ax.bar([x + j * width for x in range(len(kinds))], [analysis[key][k] for k in kinds], width, label=NAMES[key])
        ax.bar_label(bars, fontsize=7, padding=1)
    ax.set_xticks([x + width * (len(analysis) - 1) / 2 for x in range(len(kinds))],
                  ["boundary", "type", "type+boundary", "missed", "spurious", "invalid IOB"])
    ax.set_ylabel("count (test set)")
    ax.set_title("Error types per model")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "error_analysis.png", dpi=130)

    crf_ckpt = MODELS_DIR / "bilstm_crf" / "model.pt"
    if crf_ckpt.exists():
        trans = torch.load(crf_ckpt, map_location="cpu", weights_only=True)["state_dict"]["crf.transitions"].numpy()
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        lim = abs(trans).max()
        im = ax.imshow(trans, cmap="RdBu", vmin=-lim, vmax=lim)
        ax.set_xticks(range(len(LABELS)), LABELS, rotation=45)
        ax.set_yticks(range(len(LABELS)), LABELS)
        for i in range(len(LABELS)):
            for j in range(len(LABELS)):
                ax.text(j, i, f"{trans[i, j]:.1f}", ha="center", va="center", fontsize=7)
        ax.set_xlabel("to tag")
        ax.set_ylabel("from tag")
        ax.set_title("Learned CRF transition scores")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "crf_transitions.png", dpi=130)

    fig, ax = plt.subplots(figsize=(7, 4))
    for key, m in models.items():
        h = m["metrics"]["history"]
        if key == "transformer":
            ax.plot([x["epoch"] for x in h], [x["eval_f1"] for x in h], marker="o", label=NAMES[key])
        else:
            ax.plot([x["epoch"] for x in h], [x["dev_f1"] for x in h], label=NAMES[key])
    ax.set_xlabel("epoch")
    ax.set_ylabel("dev F1")
    ax.set_title("Validation F1 during training")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "learning_curves.png", dpi=130)


if __name__ == "__main__":
    main()
