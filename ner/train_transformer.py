"""Fine-tune a HuggingFace transformer (BERT / DistilBERT) for token classification.

    python -m ner.train_transformer --model bert-base-cased
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from datasets import Dataset, DatasetDict
from seqeval.metrics import classification_report, f1_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

from . import ID2LABEL, LABEL2ID, LABELS, MODELS_DIR, RESULTS_DIR
from .data import load_conll


def to_hf(data) -> DatasetDict:
    return DatasetDict({
        s: Dataset.from_dict({"tokens": [t for t, _ in sents], "ner_tags": [[LABEL2ID[x] for x in g] for _, g in sents]})
        for s, sents in data.items()
    })


def tokenize_and_align(examples, tokenizer, label_all_subwords: bool = False):
    """Sub-word tokenization + label alignment.

    WordPiece may split one word into several pieces ("Blackburn" -> "Black", "##burn").
    Only the first piece of each word keeps the word's label; the other pieces and the
    special tokens ([CLS], [SEP]) get -100 so the loss and the metrics ignore them.
    """
    enc = tokenizer(examples["tokens"], is_split_into_words=True, truncation=True)
    all_labels = []
    for i, tags in enumerate(examples["ner_tags"]):
        prev, labels = None, []
        for wid in enc.word_ids(batch_index=i):
            if wid is None:
                labels.append(-100)
            elif wid != prev:
                labels.append(tags[wid])
            else:
                tag = LABELS[tags[wid]]
                labels.append(LABEL2ID["I-" + tag[2:]] if label_all_subwords and tag != "O" else -100)
            prev = wid
        all_labels.append(labels)
    enc["labels"] = all_labels
    return enc


def align_predictions(logits, label_ids):
    preds = np.argmax(logits, axis=-1)
    true, pred = [], []
    for p_row, l_row in zip(preds, label_ids):
        keep = l_row != -100
        true.append([ID2LABEL[int(x)] for x in l_row[keep]])
        pred.append([ID2LABEL[int(x)] for x in p_row[keep]])
    return true, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bert-base-cased")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)

    data = load_conll()
    ds = to_hf(data)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenized = ds.map(lambda ex: tokenize_and_align(ex, tokenizer), batched=True, remove_columns=ds["train"].column_names)

    model = AutoModelForTokenClassification.from_pretrained(
        args.model, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID
    )

    def compute_metrics(p):
        true, pred = align_predictions(p.predictions, p.label_ids)
        return {"f1": f1_score(true, pred)}

    out_dir = MODELS_DIR / "transformer"
    targs = TrainingArguments(
        output_dir=str(MODELS_DIR / "transformer_ckpt"),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=64,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=__import__("torch").cuda.is_available(),
        logging_steps=100,
        report_to="none",
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorForTokenClassification(tokenizer),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    results = {"arch": "transformer", "base_model": args.model,
               "history": [h for h in trainer.state.log_history if "eval_f1" in h]}
    for split in ("validation", "test"):
        out = trainer.predict(tokenized[split])
        true, pred = align_predictions(out.predictions, out.label_ids)
        # Sentences truncated by the tokenizer would be shorter; pad with O so lengths match the gold data.
        gold = [tags for _, tags in data[split]]
        pred = [p + ["O"] * (len(g) - len(p)) for p, g in zip(pred, gold)]
        results[split] = classification_report(gold, pred, digits=4, output_dict=True)
        if split == "test":
            print(classification_report(gold, pred, digits=4))
            (RESULTS_DIR / "predictions").mkdir(parents=True, exist_ok=True)
            with open(RESULTS_DIR / "predictions" / "transformer_test.json", "w") as f:
                json.dump(pred, f)
    results["best_dev_f1"] = results["validation"]["micro avg"]["f1-score"]
    with open(RESULTS_DIR / "transformer_metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=float)


if __name__ == "__main__":
    main()
