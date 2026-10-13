"""Inference on free text for all trained models (used by the Gradio app)."""
from __future__ import annotations

import torch

from . import ID2LABEL, MODELS_DIR
from .data import Vocab, encode_tokens, get_spans, tokenize_with_offsets
from .models import RNNTagger

RNN_ARCHS = {"bilstm_crf": "BiLSTM + CRF", "bilstm": "BiLSTM", "lstm": "LSTM"}
TRANSFORMER_NAME = "Transformer (BERT fine-tuned)"


class RNNPredictor:
    def __init__(self, arch: str, device="cpu"):
        ckpt = torch.load(MODELS_DIR / arch / "model.pt", map_location=device, weights_only=True)
        self.vocab = Vocab.from_dict(ckpt["vocab"])
        self.model = RNNTagger(**ckpt["config"]).to(device).eval()
        self.model.load_state_dict(ckpt["state_dict"])
        self.device = device

    def tag(self, tokens: list[str]) -> list[str]:
        batch = {k: v.to(self.device) for k, v in encode_tokens(tokens, self.vocab).items()}
        return [ID2LABEL[i] for i in self.model.decode(batch)[0]]


class TransformerPredictor:
    def __init__(self, path=MODELS_DIR / "transformer", device="cpu"):
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForTokenClassification.from_pretrained(path).to(device).eval()
        self.device = device

    @torch.no_grad()
    def tag(self, tokens: list[str]) -> list[str]:
        enc = self.tokenizer(tokens, is_split_into_words=True, truncation=True, return_tensors="pt").to(self.device)
        pred = self.model(**enc).logits[0].argmax(-1).tolist()
        tags, prev = ["O"] * len(tokens), None
        for pos, wid in enumerate(enc.word_ids()):
            if wid is not None and wid != prev:
                tags[wid] = self.model.config.id2label[pred[pos]]
            prev = wid
        return tags


def available_models() -> dict[str, str]:
    """Display name -> key for every model that has been trained."""
    out = {}
    if (MODELS_DIR / "transformer" / "config.json").exists():
        out[TRANSFORMER_NAME] = "transformer"
    for key, name in RNN_ARCHS.items():
        if (MODELS_DIR / key / "model.pt").exists():
            out[name] = key
    return out


def load_predictor(key: str, device="cpu"):
    return TransformerPredictor(device=device) if key == "transformer" else RNNPredictor(key, device=device)


def extract_entities(text: str, predictor) -> list[dict]:
    """Tokenize free text, tag it and return character-level entity spans."""
    toks = tokenize_with_offsets(text)
    if not toks:
        return []
    tags = predictor.tag([t for t, _, _ in toks])
    return [
        {"entity": etype, "start": toks[s][1], "end": toks[e - 1][2], "word": text[toks[s][1]: toks[e - 1][2]]}
        for etype, s, e in get_spans(tags)
    ]
