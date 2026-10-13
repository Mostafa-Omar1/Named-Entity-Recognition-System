"""Gradio web app: type text, entities are highlighted in real time (PER / ORG / LOC / MISC).

    python app.py
"""
from functools import lru_cache

import gradio as gr
import torch

from ner import ENTITY_COLORS
from ner.predict import available_models, extract_entities, load_predictor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODELS = available_models()
if not MODELS:
    raise SystemExit("No trained models found in models/. Train one first, e.g. `python -m ner.train_rnn --arch bilstm_crf`.")

EXAMPLES = [
    "Barack Obama met Angela Merkel in Berlin to discuss the future of the European Union.",
    "Apple CEO Tim Cook announced the new iPhone at the Steve Jobs Theater in Cupertino, California.",
    "Cristiano Ronaldo scored twice as Manchester United beat Arsenal at Old Trafford on Sunday.",
    "The World Health Organization said the outbreak in West Africa was slowing, according to Reuters.",
]


@lru_cache(maxsize=None)
def get_predictor(key):
    return load_predictor(key, device=DEVICE)


def analyze(text, model_name):
    if not text or not text.strip():
        return {"text": "", "entities": []}, []
    ents = extract_entities(text, get_predictor(MODELS[model_name]))
    rows = [[e["word"], e["entity"], e["start"], e["end"]] for e in ents]
    return {"text": text, "entities": [{k: e[k] for k in ("entity", "start", "end")} for e in ents]}, rows


def compare_all(text):
    outs = []
    for name in MODELS:
        ents = extract_entities(text, get_predictor(MODELS[name])) if text and text.strip() else []
        outs.append({"text": text or "", "entities": [{k: e[k] for k in ("entity", "start", "end")} for e in ents]})
    return outs


legend = " ".join(
    f'<span style="background:{c};padding:2px 8px;border-radius:4px;margin-right:6px;color:#111"><b>{t}</b></span>'
    for t, c in ENTITY_COLORS.items()
)

with gr.Blocks(title="Named Entity Recognition") as demo:
    gr.Markdown("# Named Entity Recognition\nType or paste English text, and the detected entities are highlighted as you type.")
    gr.HTML(f"<div>{legend}</div>")
    with gr.Tab("Analyze"):
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(label="Input text", lines=6, value=EXAMPLES[0])
                model = gr.Dropdown(list(MODELS), value=list(MODELS)[0], label="Model")
            with gr.Column():
                highlighted = gr.HighlightedText(label="Entities", color_map=ENTITY_COLORS, combine_adjacent=False, show_legend=True)
                table = gr.Dataframe(headers=["Entity", "Type", "Start", "End"], label="Detected entities", interactive=False)
        gr.Examples(EXAMPLES, inputs=text)
        text.change(analyze, [text, model], [highlighted, table], show_progress="hidden", trigger_mode="always_last")
        model.change(analyze, [text, model], [highlighted, table])
        demo.load(analyze, [text, model], [highlighted, table])
    with gr.Tab("Compare models"):
        ctext = gr.Textbox(label="Input text", lines=4, value=EXAMPLES[2])
        cmp_outputs = [gr.HighlightedText(label=name, color_map=ENTITY_COLORS, show_legend=False) for name in MODELS]
        ctext.change(compare_all, ctext, cmp_outputs, show_progress="hidden", trigger_mode="always_last")
        demo.load(compare_all, ctext, cmp_outputs)

if __name__ == "__main__":
    demo.launch()
