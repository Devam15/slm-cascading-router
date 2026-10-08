# SLM Cascading Router — Self-Consistency Escalation

A small local language model answers every question. When it isn't sure of itself, the question is escalated to a bigger model.

## How it works

1. **Ask the small model 3 times.** Qwen2.5-1.5B (via Ollama) answers the same question three times with sampling on (temperature 0.8).
2. **Check if it agrees with itself.** The three answers are embedded with `all-MiniLM-L6-v2` and their average pairwise cosine similarity becomes the *consistency score*. Agreeing answers suggest confidence; disagreeing answers suggest uncertainty.
3. **Escalate if it disagrees.** If consistency < 0.75, the question goes to the big model (GPT-4.1-mini by default, or a local `qwen2.5:7b`), which answers once.
4. **Score everything.** Final answers are compared to the dataset's ground truth with embedding similarity (≥ 0.55 counts as correct), with and without routing.

## Dataset

[MedHallu](https://huggingface.co/datasets/UTAustin-AIHealth/MedHallu) (`pqa_labeled`, 1,000 expert-labelled medical Q&A rows). A 350-row subset, stratified by difficulty and hallucination category, is in `data/`. It contains the original 120-row pilot as a literal subset.

### Dataset & licence

The files in `data/` are subsets of MedHallu, redistributed under its MIT licence. MedHallu's questions are derived from PubMedQA, also MIT-licensed. If you use this data, please cite the original work:

- **MedHallu:** Pandit et al., *MedHallu: A Comprehensive Benchmark for Detecting Medical Hallucinations in Large Language Models*, 2025. [arXiv:2502.14302](https://arxiv.org/abs/2502.14302) · [GitHub](https://github.com/MedHallu/MedHallu) · [Hugging Face](https://huggingface.co/datasets/UTAustin-AIHealth/MedHallu)
- **PubMedQA:** Jin et al., *PubMedQA: A Dataset for Biomedical Research Question Answering*, EMNLP 2019. [GitHub](https://github.com/pubmedqa/pubmedqa)

## Results (350 questions)

| | |
|---|---|
| Questions escalated | 24 / 350 (6.9%) |
| Small model alone — accuracy | 74.0% |
| Small model + routing — accuracy | 76.6% |
| Wrong answers caught by escalation | 14 / 91 (15.4%) |
| Caught answers fixed by the big model | 9 of 14 |
| Correct answers broken by escalating | 0 |
| Avg consistency — correct vs. wrong answers | 0.921 vs. 0.868 |

**Takeaway:** self-consistency is a real but weak signal on medical questions. Escalating ~7% of questions catches ~15% of the errors, and routing never made a correct answer worse. Its blind spot: a model that is *consistently* wrong looks confident.

## Files

| File | What it does |
|---|---|
| `router_demo.py` | The router: sample, score consistency, escalate, evaluate |
| `dataset_prep.py` | Downloads MedHallu and builds the 120-row stratified pilot |
| `dataset_prep_extend.py` | Extends the pilot to 350 rows, keeping the original 120 |
| `data/` | The 120- and 350-row subsets |
| `results/router_results.csv` | Per-question results of the 350-row run |

## Run it

```bash
ollama pull qwen2.5:1.5b
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."   # only for the default openai backend; never commit keys
python router_demo.py
```

Set `BIG_MODEL_BACKEND = "ollama"` in `router_demo.py` to keep everything local.