"""LSTM, BiLSTM and BiLSTM-CRF sequence taggers, implemented from scratch in PyTorch.

Every token is represented as   [GloVe word embedding ; char-CNN embedding ; casing embedding]
so out-of-vocabulary words still get a meaningful vector from their characters.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharCNN(nn.Module):
    """Character-level word encoder (Ma & Hovy, 2016): embed chars -> 1D conv -> max-pool."""

    def __init__(self, n_chars: int, char_dim: int = 30, n_filters: int = 50, kernel: int = 3, dropout: float = 0.5):
        super().__init__()
        self.emb = nn.Embedding(n_chars, char_dim, padding_idx=0)
        self.conv = nn.Conv1d(char_dim, n_filters, kernel, padding=kernel // 2)
        self.drop = nn.Dropout(dropout)
        self.out_dim = n_filters

    def forward(self, chars):  # chars: (B, T, L)
        B, T, L = chars.shape
        x = self.drop(self.emb(chars.view(B * T, L)))  # (BT, L, C)
        x = self.conv(x.transpose(1, 2))  # (BT, F, L)
        x = x.masked_fill((chars.view(B * T, 1, L) == 0), -1e4)
        x = x.max(dim=2).values
        x = torch.where(x <= -1e4 + 1, torch.zeros_like(x), x)  # padded words -> 0
        return x.view(B, T, -1)


class CRF(nn.Module):
    """Linear-chain conditional random field with learned transition scores.

    Instead of choosing each tag independently (softmax), the CRF scores whole tag
    sequences:  score(y) = sum_t emission[t, y_t] + transition[y_{t-1}, y_t]
    and is trained by maximizing log p(y|x) = score(y) - logsumexp_{y'} score(y').
    Decoding uses Viterbi to find the globally best sequence, so the model can learn
    that e.g. O -> I-PER is (almost) never valid.
    """

    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags).uniform_(-0.1, 0.1))  # [from, to]
        self.start = nn.Parameter(torch.empty(num_tags).uniform_(-0.1, 0.1))
        self.end = nn.Parameter(torch.empty(num_tags).uniform_(-0.1, 0.1))

    def _score(self, em, tags, mask):  # gold path score, (B,)
        B, T, _ = em.shape
        idx = torch.arange(B, device=em.device)
        score = self.start[tags[:, 0]] + em[idx, 0, tags[:, 0]]
        for t in range(1, T):
            step = self.transitions[tags[:, t - 1], tags[:, t]] + em[idx, t, tags[:, t]]
            score = score + step * mask[:, t]
        last = tags.gather(1, (mask.sum(1).long() - 1).unsqueeze(1)).squeeze(1)
        return score + self.end[last]

    def _partition(self, em, mask):  # log Z via forward algorithm, (B,)
        alpha = self.start + em[:, 0]  # (B, K)
        for t in range(1, em.size(1)):
            nxt = torch.logsumexp(alpha.unsqueeze(2) + self.transitions + em[:, t].unsqueeze(1), dim=1)
            alpha = torch.where(mask[:, t].unsqueeze(1), nxt, alpha)
        return torch.logsumexp(alpha + self.end, dim=1)

    def neg_log_likelihood(self, em, tags, mask):
        em = em.float()
        return (self._partition(em, mask) - self._score(em, tags, mask)).mean()

    @torch.no_grad()
    def viterbi(self, em, mask) -> list[list[int]]:
        em = em.float()
        B, T, K = em.shape
        score = self.start + em[:, 0]
        history = []
        for t in range(1, T):
            total = score.unsqueeze(2) + self.transitions  # (B, K_from, K_to)
            best, idx = total.max(dim=1)
            nxt = best + em[:, t]
            score = torch.where(mask[:, t].unsqueeze(1), nxt, score)
            history.append(idx)
        score = score + self.end
        lengths = mask.sum(1).tolist()
        paths = []
        for b in range(B):
            best = score[b].argmax().item()
            path = [best]
            for hist in reversed(history[: lengths[b] - 1]):
                best = hist[b, best].item()
                path.append(best)
            paths.append(path[::-1])
        return paths


class RNNTagger(nn.Module):
    """One class, three architectures:

    - ``lstm``        : unidirectional LSTM + softmax (each tag only sees left context)
    - ``bilstm``      : bidirectional LSTM + softmax (left + right context, independent tags)
    - ``bilstm_crf``  : bidirectional LSTM + CRF output layer (joint tag-sequence decoding)
    """

    def __init__(self, arch: str, n_words: int, n_chars: int, n_casings: int, n_tags: int,
                 word_dim: int = 100, hidden: int = 200, case_dim: int = 10, dropout: float = 0.5,
                 pretrained: torch.Tensor | None = None):
        super().__init__()
        assert arch in ("lstm", "bilstm", "bilstm_crf"), arch
        self.arch = arch
        self.word_emb = nn.Embedding(n_words, word_dim, padding_idx=0)
        if pretrained is not None:
            self.word_emb.weight.data.copy_(pretrained)
        self.char_cnn = CharCNN(n_chars, dropout=dropout)
        self.case_emb = nn.Embedding(n_casings, case_dim, padding_idx=0)
        in_dim = word_dim + self.char_cnn.out_dim + case_dim
        bidir = arch != "lstm"
        # Give the unidirectional model the same total hidden size for a fair comparison.
        self.lstm = nn.LSTM(in_dim, hidden if bidir else 2 * hidden, batch_first=True, bidirectional=bidir)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(2 * hidden, n_tags)
        self.crf = CRF(n_tags) if arch == "bilstm_crf" else None

    def emissions(self, batch):
        x = torch.cat([self.word_emb(batch["words"]), self.char_cnn(batch["chars"]), self.case_emb(batch["case"])], dim=-1)
        x = self.drop(x)
        lengths = batch["mask"].sum(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        return self.proj(self.drop(out))

    def loss(self, batch):
        em = self.emissions(batch)
        if self.crf is not None:
            return self.crf.neg_log_likelihood(em, batch["tags"], batch["mask"])
        return F.cross_entropy(em[batch["mask"]], batch["tags"][batch["mask"]])

    @torch.no_grad()
    def decode(self, batch) -> list[list[int]]:
        em = self.emissions(batch)
        if self.crf is not None:
            return self.crf.viterbi(em, batch["mask"])
        pred = em.argmax(-1)
        return [pred[b, : int(n)].tolist() for b, n in enumerate(batch["mask"].sum(1))]
