"""CoNLL-2003 loading, IOB tagging utilities and vocabularies for the RNN taggers."""
from __future__ import annotations

import io
import re
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import Dataset

from . import DATA_DIR, LABEL2ID

SPLITS = {"train": "train.txt", "validation": "valid.txt", "test": "test.txt"}
CONLL_ZIP_URL = "https://data.deepai.org/conll2003.zip"

PAD, UNK = "<pad>", "<unk>"


# --------------------------------------------------------------------------- loading
def ensure_conll(data_dir: Path = DATA_DIR) -> Path:
    """Make sure the raw CoNLL-2003 files exist locally, downloading them if needed."""
    data_dir.mkdir(parents=True, exist_ok=True)
    if all((data_dir / f).exists() for f in SPLITS.values()):
        return data_dir
    print(f"Downloading CoNLL-2003 from {CONLL_ZIP_URL} ...")
    with urllib.request.urlopen(CONLL_ZIP_URL) as r:
        archive = zipfile.ZipFile(io.BytesIO(r.read()))
    for name in SPLITS.values():
        member = next(m for m in archive.namelist() if m.endswith(name))
        (data_dir / name).write_bytes(archive.read(member))
    return data_dir


def read_conll(path: Path) -> list[tuple[list[str], list[str]]]:
    """Read a CoNLL-2003 file (token POS chunk NER per line) into (tokens, ner_tags) sentences."""
    sentences, tokens, tags = [], [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("-DOCSTART-"):
            if tokens:
                sentences.append((tokens, to_iob2(tags)))
                tokens, tags = [], []
            continue
        parts = line.split()
        tokens.append(parts[0])
        tags.append(parts[-1])
    if tokens:
        sentences.append((tokens, to_iob2(tags)))
    return sentences


def load_conll(data_dir: Path = DATA_DIR) -> dict[str, list[tuple[list[str], list[str]]]]:
    data_dir = ensure_conll(data_dir)
    return {split: read_conll(data_dir / fname) for split, fname in SPLITS.items()}


# --------------------------------------------------------------------------- IOB scheme
def to_iob2(tags: list[str]) -> list[str]:
    """Convert IOB1 (original CoNLL, B- only between adjacent same-type entities) to IOB2.

    In IOB2 every entity starts with B-, which makes entity boundaries explicit. The
    conversion is idempotent, so files that are already IOB2 pass through unchanged.
    """
    out = []
    for i, tag in enumerate(tags):
        if tag.startswith("I-"):
            prev = out[i - 1] if i > 0 else "O"
            if prev == "O" or prev[2:] != tag[2:]:
                tag = "B-" + tag[2:]
        out.append(tag)
    return out


def get_spans(tags: list[str]) -> list[tuple[str, int, int]]:
    """Extract (type, start, end_exclusive) entity spans from an IOB2 sequence.

    A dangling I-X (not preceded by B-X / I-X) starts a new span, mirroring seqeval's
    lenient default behaviour.
    """
    spans, start, etype = [], None, None
    for i, tag in enumerate(list(tags) + ["O"]):
        prefix, _, t = tag.partition("-")
        continues = prefix == "I" and t == etype and start is not None
        if not continues and start is not None:
            spans.append((etype, start, i))
            start, etype = None, None
        if prefix in ("B", "I") and not continues:
            start, etype = i, t
    return spans


def invalid_transitions(tags: list[str]) -> int:
    """Count IOB2 violations: I-X that does not follow B-X or I-X."""
    bad, prev = 0, "O"
    for tag in tags:
        if tag.startswith("I-") and (prev == "O" or prev[2:] != tag[2:]):
            bad += 1
        prev = tag
    return bad


# --------------------------------------------------------------------------- tokenizer for free text
TOKEN_RE = re.compile(r"\w+(?:[-'.]\w+)*|[^\w\s]", re.UNICODE)


def tokenize_with_offsets(text: str) -> list[tuple[str, int, int]]:
    """Simple CoNLL-like word tokenizer that keeps character offsets (for highlighting)."""
    return [(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


# --------------------------------------------------------------------------- vocabularies
def normalize_word(w: str) -> str:
    """Lower-case to match the uncased GloVe vocabulary (case is recovered via casing + chars)."""
    return w.lower()


CASINGS = ["<pad>", "numeric", "mostly_numeric", "all_lower", "all_upper", "init_upper", "contains_digit", "other"]


def casing(word: str) -> int:
    digits = sum(c.isdigit() for c in word)
    if word.isdigit():
        c = "numeric"
    elif digits / max(len(word), 1) > 0.5:
        c = "mostly_numeric"
    elif word.islower():
        c = "all_lower"
    elif word.isupper():
        c = "all_upper"
    elif word[0].isupper():
        c = "init_upper"
    elif digits:
        c = "contains_digit"
    else:
        c = "other"
    return CASINGS.index(c)


@dataclass
class Vocab:
    words: list[str] = field(default_factory=lambda: [PAD, UNK])
    chars: list[str] = field(default_factory=lambda: [PAD, UNK])

    def __post_init__(self):
        self.w2i = {w: i for i, w in enumerate(self.words)}
        self.c2i = {c: i for i, c in enumerate(self.chars)}

    @classmethod
    def build(cls, train_sents, extra_words=(), min_char_freq: int = 1) -> "Vocab":
        """Word vocab = all normalized training words + `extra_words` (pre-trained words seen in
        dev/test, which only adds rows the embedding already knows - no label information leaks).
        Character vocab comes from training data only; unseen characters map to <unk>."""
        wc = Counter(normalize_word(w) for toks, _ in train_sents for w in toks)
        cc = Counter(c for toks, _ in train_sents for w in toks for c in w)
        words = [PAD, UNK] + sorted(wc)
        seen = set(words)
        words += sorted(w for w in set(extra_words) if w not in seen)
        chars = [PAD, UNK] + sorted(c for c, n in cc.items() if n >= min_char_freq)
        return cls(words, chars)

    def word_id(self, w: str) -> int:
        return self.w2i.get(normalize_word(w), 1)

    def char_ids(self, w: str) -> list[int]:
        return [self.c2i.get(c, 1) for c in w]

    def to_dict(self):
        return {"words": self.words, "chars": self.chars}

    @classmethod
    def from_dict(cls, d):
        return cls(d["words"], d["chars"])


# --------------------------------------------------------------------------- torch dataset
class NERDataset(Dataset):
    def __init__(self, sentences, vocab: Vocab):
        self.items = []
        for toks, tags in sentences:
            self.items.append(
                (
                    [vocab.word_id(w) for w in toks],
                    [vocab.char_ids(w) for w in toks],
                    [casing(w) for w in toks],
                    [LABEL2ID[t] for t in tags],
                )
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def encode_tokens(tokens: list[str], vocab: Vocab):
    """Encode a single tokenized sentence (no labels) as a batch of size 1."""
    item = ([vocab.word_id(w) for w in tokens], [vocab.char_ids(w) for w in tokens], [casing(w) for w in tokens], [0] * len(tokens))
    return collate([item])


def collate(batch, max_word_len: int = 30):
    B = len(batch)
    T = max(len(b[0]) for b in batch)
    L = min(max_word_len, max(len(c) for b in batch for c in b[1]))
    words = torch.zeros(B, T, dtype=torch.long)
    chars = torch.zeros(B, T, L, dtype=torch.long)
    case = torch.zeros(B, T, dtype=torch.long)
    tags = torch.zeros(B, T, dtype=torch.long)
    mask = torch.zeros(B, T, dtype=torch.bool)
    for i, (w, c, cs, t) in enumerate(batch):
        n = len(w)
        words[i, :n] = torch.tensor(w)
        case[i, :n] = torch.tensor(cs)
        tags[i, :n] = torch.tensor(t)
        mask[i, :n] = True
        for j, cw in enumerate(c):
            cw = cw[:L]
            chars[i, j, : len(cw)] = torch.tensor(cw)
    return {"words": words, "chars": chars, "case": case, "tags": tags, "mask": mask}
