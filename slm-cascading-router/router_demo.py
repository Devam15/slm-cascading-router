"""
Router demo: small local model + self-consistency-based escalation to a
bigger local model.

WHAT THIS DOES
1. For each question, asks the SMALL model the same question N_SAMPLES
   times (with randomness on). If its answers agree with each other,
   we trust it (low uncertainty). If they disagree, that disagreement
   is a practical stand-in for "epistemic uncertainty" (Malinin & Gales,
   2021) -- a real ensemble isn't feasible in a week, so repeated
   sampling of one model approximates the same signal.
2. When the small model disagrees with itself past a threshold, the
   question is ESCALATED to the bigger model, which answers once.
3. Every final answer (whichever model produced it) is scored against
   the dataset's Ground Truth using embedding similarity.
4. Reports: accuracy with routing vs. small-model-only baseline, and
   what fraction of questions actually got escalated.

SETUP (run on your own machine -- this needs Ollama running locally
for the SMALL model either way; the BIG model can be local or the
OpenAI API, see BIG_MODEL_BACKEND below)
  1. Install Ollama: https://ollama.com
  2. Pull the small model this script always uses locally:
       ollama pull qwen2.5:1.5b
  3. Pick a backend for the BIG model (see BIG_MODEL_BACKEND below):
       - "ollama": also run  ollama pull llama3.1:8b  (or similar)
       - "openai": no download needed, but see NOTE below.
  4. pip install ollama sentence-transformers pandas
     (add "openai" to that list too if using the openai backend)
  5. If using the openai backend, set your API key as an environment
     variable -- NEVER paste it into this file:
       export OPENAI_API_KEY="sk-..."
     (put that line in your shell, or run it before python3 below)
  6. Make sure data/medhallu_pilot_subset_350.csv exists (built by
     dataset_prep.py + dataset_prep_extend.py; already included in this repo).
  7. python3 router_demo.py

This runs on a small slice (N_QUESTIONS) of your pilot subset by
default, so it finishes in a few minutes -- raise it once you've
confirmed it works end to end.

NOTE on the openai backend: this is meant for quick testing only --
OpenAI's models are closed-source (not what your project brief asked
for) and calling them sends every escalated question to OpenAI's
servers over the internet, which works against the "keep data local"
motivation behind this whole project. Use "ollama" (fully local,
open-source) for anything you'd actually show as your project's
approach; use "openai" only to sanity-check the mechanism quickly
without a multi-GB download.
"""

import csv
import os
import time
import ollama
from sentence_transformers import SentenceTransformer, util

# ---- config ----------------------------------------------------------
SMALL_MODEL = "qwen2.5:1.5b"          # always via Ollama, local

BIG_MODEL_BACKEND = "openai"          # "ollama" (local, open-source) or "openai" (API, testing only)
BIG_MODEL_OLLAMA = "qwen2.5:7b"       # used if BIG_MODEL_BACKEND == "ollama" -- same family as the small model
BIG_MODEL_OPENAI = "gpt-4.1-mini"     # used if BIG_MODEL_BACKEND == "openai" -- non-reasoning model, supports custom temperature (gpt-5-mini and other reasoning-family models only allow the default temperature of 1)

N_SAMPLES = 3            # how many times to sample the small model
N_QUESTIONS = 350        # how many rows of the subset to run
CONSISTENCY_THRESHOLD = 0.75   # below this avg similarity -> escalate
ACCURACY_THRESHOLD = 0.55      # embedding similarity counted as "correct"
INPUT_CSV = "data/medhallu_pilot_subset_350.csv"
OUTPUT_CSV = "results/router_results.csv"
# -----------------------------------------------------------------------

print("Loading judge/embedding model (first run downloads it, ~80MB)...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

_openai_client = None
if BIG_MODEL_BACKEND == "openai":
    from openai import OpenAI
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "BIG_MODEL_BACKEND is 'openai' but OPENAI_API_KEY isn't set.\n"
            "Run:  export OPENAI_API_KEY=\"sk-...\"   in your terminal first."
        )
    _openai_client = OpenAI()


def similarity(a: str, b: str) -> float:
    emb = embedder.encode([a, b], convert_to_tensor=True)
    return float(util.cos_sim(emb[0], emb[1]))


def ask_model(model: str, question: str, temperature: float) -> str:
    """Ask the SMALL model (always local, via Ollama)."""
    resp = ollama.chat(
        model=model,
        messages=[{
            "role": "user",
            "content": f"Answer this medical question concisely and directly:\n\n{question}",
        }],
        options={"temperature": temperature},
    )
    return resp["message"]["content"].strip()


def ask_big_model(question: str, temperature: float) -> str:
    """Ask the BIG model, via whichever backend is configured above."""
    prompt = f"Answer this medical question concisely and directly:\n\n{question}"
    if BIG_MODEL_BACKEND == "openai":
        try:
            resp = _openai_client.chat.completions.create(
                model=BIG_MODEL_OPENAI,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
        except Exception as e:
            # Some OpenAI models (the "reasoning" family: o-series, some
            # GPT-5 variants) only support the default temperature (1) and
            # reject any other value outright. Rather than crash the whole
            # run if a reasoning-family model is ever configured here,
            # retry once without specifying temperature at all.
            if "temperature" in str(e).lower() and "unsupported" in str(e).lower():
                print(f"    [note] {BIG_MODEL_OPENAI} doesn't support custom temperature, "
                      f"retrying with the default...")
                resp = _openai_client.chat.completions.create(
                    model=BIG_MODEL_OPENAI,
                    messages=[{"role": "user", "content": prompt}],
                )
            else:
                raise
        return resp.choices[0].message.content.strip()
    else:
        return ask_model(BIG_MODEL_OLLAMA, question, temperature)


def self_consistency_score(samples: list[str]) -> float:
    """Average pairwise similarity across N samples. 1.0 = perfect agreement."""
    if len(samples) < 2:
        return 1.0
    pairs = []
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            pairs.append(similarity(samples[i], samples[j]))
    return sum(pairs) / len(pairs)


def run_router(question: str) -> dict:
    small_samples = [ask_model(SMALL_MODEL, question, temperature=0.8) for _ in range(N_SAMPLES)]
    consistency = self_consistency_score(small_samples)
    baseline_answer = small_samples[0]  # what a no-routing pipeline would have used

    if consistency < CONSISTENCY_THRESHOLD:
        final_answer = ask_big_model(question, temperature=0.2)
        escalated = True
    else:
        final_answer = baseline_answer
        escalated = False

    return {
        "final_answer": final_answer,
        "baseline_answer": baseline_answer,
        "consistency": consistency,
        "escalated": escalated,
    }


def main():
    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[:N_QUESTIONS]

    results = []
    for i, row in enumerate(rows, 1):
        question = row["Question"]
        ground_truth = row["Ground Truth"]
        print(f"[{i}/{len(rows)}] {question[:70]}...")

        t0 = time.time()
        out = run_router(question)
        elapsed = time.time() - t0

        routed_score = similarity(out["final_answer"], ground_truth)
        baseline_score = similarity(out["baseline_answer"], ground_truth)

        results.append({
            "question": question,
            "difficulty": row.get("Difficulty Level", ""),
            "consistency": round(out["consistency"], 3),
            "escalated": out["escalated"],
            "routed_score": round(routed_score, 3),
            "baseline_score": round(baseline_score, 3),
            "routed_correct": routed_score >= ACCURACY_THRESHOLD,
            "baseline_correct": baseline_score >= ACCURACY_THRESHOLD,
            "seconds": round(elapsed, 1),
        })
        print(f"    escalated={out['escalated']}  consistency={out['consistency']:.2f}  "
              f"routed_score={routed_score:.2f}  baseline_score={baseline_score:.2f}")

    # ---- summary ----
    n = len(results)
    n_escalated = sum(r["escalated"] for r in results)
    routed_acc = sum(r["routed_correct"] for r in results) / n
    baseline_acc = sum(r["baseline_correct"] for r in results) / n

    print("\n" + "=" * 60)
    print(f"Questions run:            {n}")
    print(f"Escalation rate:          {n_escalated}/{n} ({100 * n_escalated / n:.0f}%)")
    print(f"Small-model-only accuracy: {100 * baseline_acc:.0f}%")
    print(f"Small-model+routing acc:   {100 * routed_acc:.0f}%")
    print("=" * 60)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved per-question results to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
