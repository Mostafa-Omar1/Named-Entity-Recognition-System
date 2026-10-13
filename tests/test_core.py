"""Sanity checks: IOB utilities and the from-scratch CRF against brute-force enumeration.

    python -m pytest tests
"""
import itertools

import torch

from ner.data import get_spans, invalid_transitions, to_iob2
from ner.models import CRF


def test_iob1_to_iob2():
    assert to_iob2(["I-PER", "I-PER", "O", "I-LOC", "B-LOC"]) == ["B-PER", "I-PER", "O", "B-LOC", "B-LOC"]
    assert to_iob2(["B-ORG", "I-ORG"]) == ["B-ORG", "I-ORG"]


def test_spans_and_transitions():
    tags = ["B-PER", "I-PER", "O", "I-LOC", "B-ORG", "B-ORG", "I-ORG"]
    assert get_spans(tags) == [("PER", 0, 2), ("LOC", 3, 4), ("ORG", 4, 5), ("ORG", 5, 7)]
    assert invalid_transitions(tags) == 1


def _brute_force(crf, em, length):
    K = em.size(-1)
    scores = {}
    for path in itertools.product(range(K), repeat=length):
        s = crf.start[path[0]] + em[0, path[0]]
        for t in range(1, length):
            s = s + crf.transitions[path[t - 1], path[t]] + em[t, path[t]]
        scores[path] = s + crf.end[path[-1]]
    return scores


def test_crf_matches_brute_force():
    torch.manual_seed(0)
    K, T = 3, 4
    crf = CRF(K)
    em = torch.randn(2, T, K)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.bool)
    tags = torch.tensor([[0, 1, 2, 1], [2, 2, 0, 0]])
    for b in range(2):
        n = int(mask[b].sum())
        scores = _brute_force(crf, em[b], n)
        log_z = torch.logsumexp(torch.stack(list(scores.values())), 0)
        gold = scores[tuple(tags[b, :n].tolist())]
        nll = crf.neg_log_likelihood(em[b: b + 1], tags[b: b + 1], mask[b: b + 1])
        assert torch.allclose(nll, log_z - gold, atol=1e-5)
        assert crf.viterbi(em[b: b + 1], mask[b: b + 1])[0] == list(max(scores, key=lambda p: scores[p]))
