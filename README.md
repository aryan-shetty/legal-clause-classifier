# Legal Clause Classifier — Fine-Tuned Small Model vs. Frontier LLM

A **66M-parameter model, fine-tuned on 1.4% of its weights with LoRA**, classifies
contract clauses into 100 types — and **beats a zero-shot frontier LLM by 15 accuracy
points while running ~300x faster and effectively free per inference.**

## The question

When you need to classify text into a fixed set of categories, should you **prompt a big
LLM** or **fine-tune a small model**? This project answers it empirically on a real,
hard task: classifying legal contract clauses (LEDGAR, 100 clause types) — comparing a
fine-tuned DistilBERT against zero-shot Claude Haiku on the same test clauses.

## Result

**Fine-tuned model (full 10K-clause test set):** 84.3% accuracy · **0.750 macro-F1** · 0.834 weighted-F1

**Head-to-head on 500 identical test clauses:**

| Metric          | Fine-tuned (DistilBERT+LoRA) | Zero-shot Claude Haiku |
|-----------------|-----------------------------:|-----------------------:|
| Accuracy        | **84.2%**                    | 68.8%                  |
| Macro-F1        | **0.701**                    | 0.507                  |
| Weighted-F1     | **0.828**                    | 0.662                  |
| Latency/clause  | **~2 ms** (local GPU)        | 652 ms (API)           |
| Cost/clause     | **~$0** (local)              | $0.0008                |

The fine-tuned model wins on **every axis**: +15.4 accuracy points, +0.19 macro-F1,
~300x faster, and free per inference after a one-time ~17-minute training run.

## Why fine-tuning wins here

- **The macro-F1 gap (0.701 vs 0.507) is the real story.** LEDGAR has a **138x class
  imbalance** (the most common clause type appears 3,167 times, the rarest just 23), so
  macro-F1 — which weights all 100 classes equally — is the honest metric. The zero-shot
  LLM has never seen the label distribution and struggles most on rare clause types;
  the fine-tuned model has learned them. Accuracy alone would understate the gap.
- **It's a fixed-label classification task**, which is exactly where a fine-tuned small
  model reliably beats a general LLM — the model learns the 100 categories rather than
  guessing from a prompt.
- **Efficiency compounds the win:** at production scale, ~300x lower latency and near-zero
  marginal cost matter as much as accuracy.

*(Note: Haiku returned a valid label 100% of the time — no formatting failures — so the
gap is purely classification accuracy, not parsing.)*

## How it works

1. **Data** — LEDGAR (LexGLUE): 60K train / 10K val / 10K test contract clauses, 100 types.
2. **Preprocess** — tokenize at 384 tokens (covers ~95% of clauses without truncation).
3. **Fine-tune** — DistilBERT (66M params) with a 100-way head, wrapped in **LoRA**
   (rank 16) — only **962K / 68M params (1.4%)** are trained. 4 epochs, ~17 min on one RTX 5060 Ti.
4. **Evaluate** — macro/weighted-F1 + accuracy on the held-out test set.
5. **Benchmark** — zero-shot Claude Haiku on 500 identical test clauses (all 100 labels
   in the prompt), measuring accuracy, macro-F1, cost, and latency.
6. **Compare** — score both models on the same clauses for a fair head-to-head.

## Key decisions

- **Small, general base model (DistilBERT), not legal-BERT** — the point is that a *tiny,
  general* model beats a frontier LLM; a domain-pretrained model would score higher but
  weaken that narrative. (Legal-BERT is a natural next step for max accuracy.)
- **LoRA over full fine-tuning** — trains 1.4% of params, far less memory and time, with
  competitive results.
- **Macro-F1 as headline** — the imbalance-aware metric; accuracy would flatter any model
  that just predicts common clauses.
- **Same 500 clauses for both models** — a fair comparison isolates the model as the only variable.

## Stack

Python · Hugging Face Transformers · PEFT/LoRA · PyTorch (CUDA) · scikit-learn · Anthropic API

## Run it

```bash
python -m venv venv && venv\Scripts\activate       # Windows
pip install -r requirements.txt
echo ANTHROPIC_API_KEY=sk-ant-... > .env           # for the Haiku benchmark
jupyter notebook legal_clause_classifier.ipynb     # run cells top to bottom
```

## Limitations & next steps

A scoped study: one base model, one dataset, a single training run (no hyperparameter
sweep or repeated trials for confidence intervals), and a 500-clause benchmark sample.
Natural extensions: compare against legal-domain models (Legal-BERT) and larger LLMs,
add few-shot LLM baselines, and tune hyperparameters for maximum macro-F1.