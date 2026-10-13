# Model comparison (CoNLL-2003 test set, seqeval entity-level metrics)

| Model | Precision | Recall | F1 | PER F1 | ORG F1 | LOC F1 | MISC F1 | Dev F1 |
|---|---|---|---|---|---|---|---|---|
| LSTM | 0.8079 | 0.8465 | **0.8267** | 0.9243 | 0.7555 | 0.8522 | 0.7130 | 0.8852 |
| BiLSTM | 0.8872 | 0.8964 | **0.8918** | 0.9517 | 0.8557 | 0.9124 | 0.7920 | 0.9357 |
| BiLSTM + CRF | 0.9007 | 0.8947 | **0.8977** | 0.9529 | 0.8698 | 0.9170 | 0.7898 | 0.9419 |
| BERT (fine-tuned) | 0.9069 | 0.9191 | **0.9129** | 0.9631 | 0.8969 | 0.9312 | 0.7970 | 0.9480 |

## Per-entity precision / recall / F1

**LSTM**

| Entity | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| PER | 0.9277 | 0.9208 | 0.9243 | 1617 |
| ORG | 0.7615 | 0.7495 | 0.7555 | 1661 |
| LOC | 0.8032 | 0.9077 | 0.8522 | 1668 |
| MISC | 0.6721 | 0.7593 | 0.7130 | 702 |

**BiLSTM**

| Entity | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| PER | 0.9597 | 0.9437 | 0.9517 | 1617 |
| ORG | 0.8676 | 0.8441 | 0.8557 | 1661 |
| LOC | 0.8895 | 0.9365 | 0.9124 | 1668 |
| MISC | 0.7691 | 0.8162 | 0.7920 | 702 |

**BiLSTM + CRF**

| Entity | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| PER | 0.9622 | 0.9437 | 0.9529 | 1617 |
| ORG | 0.8965 | 0.8447 | 0.8698 | 1661 |
| LOC | 0.8973 | 0.9376 | 0.9170 | 1668 |
| MISC | 0.7821 | 0.7977 | 0.7898 | 702 |

**BERT (fine-tuned)**

| Entity | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| PER | 0.9664 | 0.9598 | 0.9631 | 1617 |
| ORG | 0.8851 | 0.9091 | 0.8969 | 1661 |
| LOC | 0.9332 | 0.9293 | 0.9312 | 1668 |
| MISC | 0.7710 | 0.8248 | 0.7970 | 702 |

## Error analysis: boundaries and IOB validity

| Model | Correct | Boundary err. | Type err. | Type+boundary err. | Missed | Spurious | Invalid IOB transitions |
|---|---|---|---|---|---|---|---|
| LSTM | 4781 | 304 | 401 | 245 | 96 | 187 | 313 |
| BiLSTM | 5063 | 123 | 270 | 105 | 119 | 146 | 72 |
| BiLSTM + CRF | 5053 | 88 | 278 | 67 | 133 | 124 | 2 |
| BERT (fine-tuned) | 5191 | 87 | 225 | 85 | 67 | 136 | 49 |

21 test sentences are tagged perfectly by BiLSTM + CRF while the plain BiLSTM makes a boundary error or an invalid IOB transition. Examples are in [crf_examples.html](crf_examples.html).

- *Syria - Nader Jokhadar 7*
  - BiLSTM: [Syria]LOC [Nader]ORG [Jokhadar]PER
  - BiLSTM + CRF: [Syria]LOC [Nader Jokhadar]PER
- *Dutch forward Reggie Blinker had his indefinite suspension lifted by FIFA on Friday and was set to make his Sheffield Wednesday comeback against Liverpool on Saturday .*
  - BiLSTM: [Dutch]MISC [Reggie Blinker]PER [FIFA]ORG [Sheffield]ORG [Liverpool]ORG
  - BiLSTM + CRF: [Dutch]MISC [Reggie Blinker]PER [FIFA]ORG [Sheffield Wednesday]ORG [Liverpool]ORG
- *Basketball Association teams after games played on Thursday*
  - BiLSTM: [Association]ORG
  - BiLSTM + CRF: [Basketball Association]ORG
- *NY JETS AT NEW ENGLAND*
  - BiLSTM: [NY]ORG [JETS]ORG [NEW ENGLAND]LOC
  - BiLSTM + CRF: [NY JETS]ORG [NEW ENGLAND]LOC
- *SOCCER SHOWCASE-BETTING ON REAL MADRID V BARCELONA .*
  - BiLSTM: [REAL]ORG [MADRID]ORG [BARCELONA]ORG
  - BiLSTM + CRF: [REAL MADRID]ORG [BARCELONA]ORG
