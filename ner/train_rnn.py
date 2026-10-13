"""Train an LSTM / BiLSTM / BiLSTM-CRF tagger on CoNLL-2003 with GloVe + char-CNN features.

    python -m ner.train_rnn --arch bilstm_crf
"""
from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np
import torch
from seqeval.metrics import classification_report, f1_score
from torch.utils.data import DataLoader

from . import ID2LABEL, LABELS, MODELS_DIR, RESULTS_DIR
from .data import CASINGS, NERDataset, Vocab, collate, load_conll, normalize_word
from .embeddings import GLOVE_DIM, build_matrix, load_glove
from .models import RNNTagger


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def predict(model, loader, device) -> list[list[str]]:
    model.eval()
    preds = []
    for batch in loader:
        preds += [[ID2LABEL[i] for i in seq] for seq in model.decode(to_device(batch, device))]
    return preds


def build_vocab_and_embeddings(data, glove_top_n: int):
    """Vocab = training words + dev/test words known to GloVe + the `glove_top_n` most frequent
    GloVe words, so free text at inference time (the app) also gets pre-trained vectors."""
    all_words = {normalize_word(w) for split in data.values() for toks, _ in split for w in toks}
    glove = load_glove(all_words, top_n=glove_top_n)
    vocab = Vocab.build(data["train"], extra_words=glove.keys())
    matrix, hits = build_matrix(vocab.words, glove)
    print(f"vocab={len(vocab.words)} words ({hits} initialised from GloVe), {len(vocab.chars)} chars")
    return vocab, matrix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["lstm", "bilstm", "bilstm_crf"], required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=200)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--word-dropout", type=float, default=0.05, help="prob. of replacing a word by <unk> while training")
    ap.add_argument("--glove-top-n", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_conll()
    vocab, matrix = build_vocab_and_embeddings(data, args.glove_top_n)

    loaders = {
        s: DataLoader(NERDataset(data[s], vocab), batch_size=args.batch_size, shuffle=(s == "train"), collate_fn=collate)
        for s in data
    }
    gold = {s: [tags for _, tags in data[s]] for s in data}

    config = dict(arch=args.arch, n_words=len(vocab.words), n_chars=len(vocab.chars), n_casings=len(CASINGS),
                  n_tags=len(LABELS), word_dim=GLOVE_DIM, hidden=args.hidden, dropout=args.dropout)
    model = RNNTagger(**config, pretrained=torch.from_numpy(matrix)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
    print(f"{args.arch}: {sum(p.numel() for p in model.parameters()):,} parameters on {device}")

    out_dir = MODELS_DIR / args.arch
    out_dir.mkdir(parents=True, exist_ok=True)
    best_f1, bad_epochs, history = -1.0, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        for batch in loaders["train"]:
            batch = to_device(batch, device)
            # Word dropout: teach the model what to do with <unk> (OOV) words - rely on chars/casing.
            drop = (torch.rand_like(batch["words"], dtype=torch.float) < args.word_dropout) & batch["mask"]
            batch["words"] = batch["words"].masked_fill(drop, 1)
            loss = model.loss(batch)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item()
        dev_f1 = f1_score(gold["validation"], predict(model, loaders["validation"], device))
        sched.step(dev_f1)
        history.append({"epoch": epoch, "train_loss": total / len(loaders["train"]), "dev_f1": dev_f1})
        print(f"epoch {epoch:2d} loss {history[-1]['train_loss']:.4f} dev F1 {dev_f1:.4f} ({time.time() - t0:.0f}s)")
        if dev_f1 > best_f1:
            best_f1, bad_epochs = dev_f1, 0
            torch.save({"config": config, "state_dict": model.state_dict(), "vocab": vocab.to_dict()}, out_dir / "model.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("early stopping")
                break

    ckpt = torch.load(out_dir / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    results = {"arch": args.arch, "best_dev_f1": best_f1, "history": history}
    RESULTS_DIR.mkdir(exist_ok=True)
    for split in ("validation", "test"):
        preds = predict(model, loaders[split], device)
        results[split] = classification_report(gold[split], preds, digits=4, output_dict=True)
        if split == "test":
            print(classification_report(gold[split], preds, digits=4))
            (RESULTS_DIR / "predictions").mkdir(exist_ok=True)
            with open(RESULTS_DIR / "predictions" / f"{args.arch}_test.json", "w") as f:
                json.dump(preds, f)
    with open(RESULTS_DIR / f"{args.arch}_metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=float)


if __name__ == "__main__":
    main()
