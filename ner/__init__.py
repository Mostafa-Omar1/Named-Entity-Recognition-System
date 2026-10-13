"""Named Entity Recognition system: LSTM / BiLSTM / BiLSTM-CRF from scratch + fine-tuned transformer."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "conll2003"
EMB_DIR = ROOT / "data" / "embeddings"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

ENTITY_TYPES = ["PER", "ORG", "LOC", "MISC"]
LABELS = ["O"] + [f"{p}-{t}" for t in ENTITY_TYPES for p in ("B", "I")]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

# Colors shared by the Gradio app and the HTML renderer.
ENTITY_COLORS = {"PER": "#ff7f7f", "ORG": "#7fb3ff", "LOC": "#7fdf9f", "MISC": "#ffcf6f"}
