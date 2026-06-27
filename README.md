# SLM vs LLM: Finding the Breakeven Point

This repo holds my final project for [course/semester — fill this in], where I tried to answer a question that I think actually matters for anyone deploying NLP models in the real world: **how much labeled data does it take before fine-tuning a small model beats few-shot prompting a large one?**

I call this the **Breakeven Point** — the number of training examples at which a fine-tuned DistilBERT matches or beats few-shot Mistral-7B-Instruct on the same task.

## Why I cared about this

If you're running a million inference requests a day, prompting a 7B parameter model for every single one gets expensive fast. The obvious alternative is to fine-tune something small and cheap like DistilBERT — but that only pays off once you've collected enough labeled data to make it competitive. Nobody really gives a concrete number for "enough," so I decided to measure it myself across a few different task types instead of just assuming it generalizes from one.

## What I actually did

I ran DistilBERT-base-uncased through full fine-tuning at training sizes ranging from 50 up to 1200 examples, and compared it against Mistral-7B-Instruct-v0.2 under six different few-shot configurations (k = 12 to 96 demonstrations). I tested this across four tasks that are pretty different from each other on purpose:

- **GoEmotions** — 12-class emotion recognition (informal Reddit text, subtle/overlapping classes)
- **AG News** — 4-class topic classification (clean, well-represented in pretraining)
- **SNLI** — 3-class natural language inference (requires reasoning over sentence *pairs*)
- **CoLA** — binary grammatical acceptability judgment

Each task was picked to stress a different variable — label count, how much an LLM already "knows" about the task from pretraining, and whether the task needs compositional reasoning vs. straightforward classification.

## What I found

The short version: **label count barely matters, prior LLM knowledge matters a lot.**

| Task | Classes | Mistral F1 (few-shot) | Breakeven (n*) |
|---|---|---|---|
| GoEmotions | 12 | 0.370 | ≈ 450 |
| AG News | 4 | 0.870 | > 400 (not reached in tested range) |
| SNLI | 3 | 0.785 | > 600 (not reached) |
| CoLA | 2 | 0.720 | > 1000 (not reached) |

GoEmotions has the *most* classes but the *lowest* breakeven, because Mistral genuinely struggles with fine-grained emotional nuance — its few-shot ceiling is low, so DistilBERT doesn't have to climb very far to catch it. CoLA has the *fewest* classes but never catches up in the range I tested, because grammatical acceptability is something Mistral is already pretty strong at, and DistilBERT seems to hit a capacity plateau around F1 ≈ 0.63 regardless of how much more data I gave it.

I also ran a sensitivity analysis on how many few-shot examples Mistral actually needs before it stops improving (it plateaus around k=48 on GoEmotions), and tested whether the breakeven pattern holds up if you swap in different emotion subsets (it doesn't hold perfectly — both a "positive" and "negative" emotion subset needed more data than the original mixed 12-class set).

## My rough takeaway

If you've got fewer than ~150 labeled examples, just prompt a large model — fine-tuning isn't worth it yet. Somewhere around 300–500 examples, fine-tuning starts to make sense *if* your task is something the LLM doesn't already have strong priors for. If your task is something LLMs are typically good at out of the box (topic classification, logical inference, grammar judgments), be prepared to need 1000+ examples before fine-tuning actually wins — and on some tasks, more data alone might not be enough; you may need a stronger base model than DistilBERT.

## What's in this repo

```
slm-vs-llm/
├── README.md
├── paper/
│   ├── NLP_Final_Paper.pdf          # Final write-up with full results, figures, and analysis
│   └── Project_Proposal.pdf          # Original project proposal/scope
├── notebooks/
│   ├── NLP_Final_Project.ipynb       # Core experiment: GoEmotions + AG News pipeline
│   └── NLP_All_Tasks_Complete.ipynb  # Full pipeline across all 4 tasks, all sweeps
└── presentation/
    └── NLP_Final_Presentation.pptx   # Slide deck summary
```

## Methodology, briefly

- **Fine-tuned model:** DistilBERT-base-uncased, classification head, LR 2e-5, batch size 16, up to 5 epochs with early stopping (patience=1) on validation Macro F1.
- **Few-shot model:** Mistral-7B-Instruct-v0.2, stratified demonstrations sampled per trial, greedy decoding, no weight updates.
- **Metric:** Macro F1 throughout, to handle class imbalance fairly.
- **Trials:** 4 trials per config (2 for extended sweeps) with different random seeds, mean ± std reported.

## Limitations (being upfront about these)

I evaluated on fast subsets of 90–120 stratified examples per task due to GPU constraints on Google Colab — not full test sets. That introduces more sampling noise than I'd like, and it shows up clearly in SNLI's results, where standard deviations got as high as 0.10–0.12. I also only tested one SLM (DistilBERT) and one LLM (Mistral-7B), so I can't say how much of this is specific to that pairing versus a general pattern.

## Where this could go next

- Try parameter-efficient fine-tuning (LoRA) — could shift the breakeven points significantly since it changes how much training signal is actually needed
- Test with other LLMs to see how much of Mistral's "prior knowledge advantage" is from its specific instruction tuning vs. raw scale
- Build a small calculator/tool that takes your task type, annotation budget, and expected inference volume, and spits out a recommendation

## Citation / Authors

Tathya Shah ([tts49@scarletmail.rutgers.edu](mailto:tts49@scarletmail.rutgers.edu)) and Dev Shah ([djs525@scarletmail.rutgers.edu](mailto:djs525@scarletmail.rutgers.edu)), Rutgers University.

If you use any of this, feel free to reach out or just link back to the repo.
