# Alloy — Project Specification
### Data-Centric Fine-Tuning Ablation Pipeline

**Version 1.0**

This document specifies the Alloy project completely enough that any capable LLM or engineer can implement it from this document alone, with zero additional context. It is intentionally code-free — implementation happens phase by phase, on request, after this document is agreed on.

---

## 1. Project Summary

Alloy is a controlled ablation pipeline that answers one question with quantified evidence: **which data curation decisions actually improve a fine-tuned language model, and by how much?**

Rather than fine-tuning a model once on a fixed dataset and eyeballing outputs, Alloy trains the same base model four times, changing exactly one data-curation variable at a time, and scores every resulting checkpoint against a single frozen held-out evaluation set. The deliverable is not a chatbot — it is a defensible, quantified answer to a real data-centric AI question, backed by a reproducible pipeline.

---

## 2. Goals and Non-Goals

**Goals:**
- Build a reproducible, config-driven, idempotent data pipeline for curating a technical Q&A corpus
- Train a lightweight quality classifier from noisy community signals (upvotes, accepted-answer flags)
- Implement embedding-based near-duplicate detection
- Run continued pretraining with a general-purpose replay buffer to mitigate catastrophic forgetting
- Run supervised fine-tuning (SFT) with correct loss masking
- Run DPO for preference alignment
- Execute a controlled 4-variant ablation study, isolating one curation decision per variant
- Produce a quantified, written comparison of the four variants against a fixed held-out evaluation set

**Non-goals (explicitly out of scope):**
- No reinforcement learning of any kind (no PPO, no GRPO, no reward-model rollouts) — DPO is the only preference-alignment method used
- No multi-GPU or distributed training
- No production deployment, load-bearing API, or public-facing service
- Not chasing state-of-the-art benchmark performance — the deliverable is the ablation result, not a leaderboard entry
- No architecture research — the base model is used as-is; only the data pipeline and training regime are varied

---

## 3. Environment and Constraints

**Local development machine:** NVIDIA GTX 1650, 4GB VRAM, no tensor cores (CUDA compute capability 7.5). Used exclusively for writing and smoke-testing code — tiny data slices, a handful of training steps, fast iteration. **Not** used for real training runs.

**Real training compute:** Google Colab free tier (T4, 16GB VRAM) and/or Kaggle free tier (P100 or T4, 30 GPU-hours/week quota). All four ablation variants' real training runs happen here.

**Base model:** Qwen2.5-1.5B-Instruct. (Swappable for Llama-3.2-1B/3B-Instruct — the pipeline must not hardcode assumptions that break this swap.)

**Domain corpus:** Stack Exchange data dump for `stats.stackexchange.com` (Cross Validated). Swappable for `ai.stackexchange.com` or `datascience.stackexchange.com` without pipeline redesign.

**Core libraries:**
- PyTorch — underlying tensor/training framework
- Hugging Face `datasets` — data loading and manipulation
- Unsloth — continued pretraining, SFT, and DPO; chosen specifically for its memory efficiency on constrained/free-tier GPUs
- sentence-transformers — embeddings for both the quality classifier and deduplication
- scikit-learn — the lightweight quality classifier (logistic regression or gradient-boosted trees)
- faiss-cpu or datasketch (MinHash LSH) — near-duplicate detection
- Weights & Biases (free tier) — experiment tracking, logged automatically per run
- FastAPI + SQLite — lightweight lineage tracking (which config produced which dataset version produced which checkpoint) and a minimal serving endpoint for the final model

---

## 4. High-Level Architecture

Nine sequential pipeline stages, each **idempotent** and **independently re-runnable**:

1. **Ingest** — parse the raw Stack Exchange XML dump into structured Q&A pairs with metadata
2. **Eval carve-out** — reserve a held-out set before any curation decision is made
3. **Weak-label** — derive quality labels for the classifier from score/accepted-answer heuristics
4. **Train classifier** — train the lightweight quality classifier on weak labels
5. **Dedup** — run embedding-based near-duplicate detection across the corpus
6. **Assemble variants** — combine filters into the four ablation conditions (V0–V3)
7. **Train each variant** — continued pretraining → SFT → DPO, per variant
8. **Evaluate** — score every checkpoint against the identical frozen held-out set
9. **Report** — produce the ablation comparison

*Note: Stages 4 (train classifier) and 5 (dedup) both consume the same embeddings, computed once and cached to `data/interim/embeddings.npy` + `data/interim/embeddings_index.json`, rather than each stage recomputing them independently.*

Every stage reads from and writes to versioned, named artifacts (not in-memory hand-offs), so any single stage can be re-run when one upstream decision changes, without re-running the whole pipeline.

Each experiment run is driven by a single YAML config specifying: which curation filters are active, hyperparameters, base model, and random seed. A SQLite table records, for every run: config hash → dataset version produced → checkpoint produced → eval metrics. This is the pipeline's audit trail and what makes the ablation defensible rather than anecdotal.

---

## 5. Data Specification

**Source:** the official Stack Exchange data dump (`Posts.xml` at minimum; `Users.xml`/`Votes.xml` optional for richer signals later).

**Extraction fields per Q&A pair:** question text, accepted answer text (if one exists), question score, answer score, accepted-answer flag, tags, creation timestamp, post IDs (for traceability back to source).

**Held-out evaluation set:** 500–800 Q&A pairs, reserved from the full corpus **before any other pipeline stage touches the data**, stratified by score bucket (so the eval set isn't accidentally skewed toward only high- or only low-quality examples). This set is frozen for the entire project — never included in any training variant, never used to tune the classifier, never used to tune the dedup threshold. **Ordering matters here: carving this out first is what makes every later comparison valid.**

---

## 6. Ablation Design

Four variants, each isolating exactly one additional curation decision over the previous one. All variants share: base model, random seed, training steps, and every hyperparameter not under test.

| Variant | Quality filter | Dedup | Replay buffer |
|---|---|---|---|
| **V0 — Baseline** | ✗ | ✗ | ✗ |
| **V1** | ✓ | ✗ | ✗ |
| **V2** | ✓ | ✓ | ✗ |
| **V3** | ✓ | ✓ | ✓ |

**Isolation logic:** V1 vs. V0 isolates the effect of quality filtering. V2 vs. V1 isolates the effect of deduplication. V3 vs. V2 isolates the effect of the replay buffer / forgetting mitigation. This structure is what turns four fine-tunes into an actual experiment rather than four unrelated data points.

---

## 7. Quality Classifier Specification

**Weak labeling heuristic (revised post-Phase 0):** since ingestion retains only questions with an accepted answer, the acceptance flag is constant across the pool and cannot serve as a labeling signal. The heuristic uses combined score (question_score + answer_score) alone: top-quartile combined score → positive label, bottom-quartile → negative label, middle 50% excluded as ambiguous.

**Features:** sentence-transformer embeddings of the combined question+answer text.

**Model:** logistic regression or gradient-boosted trees (scikit-learn). Deliberately not a fine-tuned transformer classifier — the classifier is a means to an end in this project, and a heavy model here would be a scope trap.

**Output:** a quality score per Q&A pair. A threshold (chosen via validation on a small held-out slice of the *weak-labeled* data — distinct from the frozen eval set) determines inclusion/exclusion for V1 and beyond.

**Caveat to document explicitly in the Phase 5 write-up:** the weak labels are derived from popularity/acceptance signals, which correlate with but are not identical to quality — a genuinely correct but unpopular answer could be mislabeled. This is a real limitation, not an implementation bug, and should be stated plainly rather than hidden.

---

## 8. Deduplication Specification

Embedding-based near-duplicate detection: compute sentence-transformer embeddings for each Q&A pair, flag pairs above a cosine-similarity threshold as near-duplicates, and keep one representative (the higher-scored one) per duplicate cluster. MinHash-based LSH is an acceptable, faster alternative if embedding-based comparison proves too slow across the full corpus on free-tier compute — this is an implementation-time decision, not an architectural one, and should be documented either way.

---

## 9. Training Pipeline (applied identically per variant, minus the variant's active filters)

**Stage A — Continued pretraining (domain-adaptive):** via Unsloth, LoRA applied to attention and MLP projection layers. For V3 specifically, also apply LoRA to the embedding and output (`lm_head`) layers with a smaller learning rate, since domain vocabulary shift is best captured there. For V3, training batches mix in general-purpose text from a FineWeb-Edu slice at a tuned ratio (start near 10–15% and treat the exact ratio as a small tunable, not a fixed constant).

**Stage B — Supervised fine-tuning (SFT):** loss computed only on response tokens, never on prompt tokens. This is a common, silent failure point — verify explicitly (not just assume) that the loss mask is applied correctly before trusting any downstream results.

**Stage C — DPO (preference tuning):** preference pairs constructed from the same corpus — accepted answer as the "chosen" response, a lower-scored answer to the same question as the "rejected" response. Pairs must be drawn only from training data, never from the frozen held-out set.

---

## 10. Evaluation Harness

**Built and frozen in Phase 2 — before any curation decision is finalized in Phase 3/4.** This ordering is a hard requirement: an eval harness designed after seeing curation results is no longer a fair judge of those results.

**Metrics, applied identically to every variant's checkpoint:**
- **Perplexity** on the frozen held-out set — measures domain fit
- **LLM-as-judge pairwise comparison** — for a sample of held-out questions, compare each variant's answer against every other variant's, blind, and tally preferences
- **General-capability check** — a lightweight MMLU-style subset (a few hundred questions spanning unrelated domains), run against every checkpoint, specifically to verify V3's replay buffer measurably preserved general capability rather than just theoretically should have

---

## 11. Reproducibility Requirements

- Fixed random seed, identical across all four variants
- Deterministic evaluation wherever the underlying tools allow it
- Checkpoint/resume logic built in from Phase 0 — not retrofitted later — given free-tier session time limits
- Every run driven by a config file, never by hardcoded script arguments

---

## 12. Phased Roadmap

### Phase 0 — Environment, Ingestion, Eval Carve-Out
**Objective:** working environment; raw data parsed into structured Q&A pairs; held-out eval set carved out and frozen before anything else touches the data.
**Checkpoint (must pass before Phase 1):** the held-out set exists as a versioned artifact, is stratified by score bucket, and is demonstrably excluded from every subsequent data-processing step by construction — not by convention.

### Phase 1 — Quality Classifier and Deduplication
**Objective:** working weak-labeling heuristic; trained quality classifier; working deduplication pass; both producing versioned, reusable artifacts.
**Checkpoint:** classifier precision/recall reported against a held-out slice of the *weak-labeled* data (distinct from the frozen eval set); dedup pass demonstrably catches a set of manually-identified known duplicates.

### Phase 2 — Evaluation Harness
**Objective:** perplexity scoring, LLM-as-judge pairwise comparison, and the general-capability subset, all built and run once against the base (untrained) model to confirm they work end-to-end.
**Checkpoint:** harness produces a complete, sane metrics report for the untrained base model before any fine-tuning has happened. This is the frozen harness — no further changes to it after this checkpoint passes.

### Phase 3 — V0 Baseline, Full Pipeline End-to-End
**Objective:** prove the full mechanical pipeline works — continued pretraining → SFT → DPO → evaluation — on the naive, unfiltered baseline.
**Checkpoint:** V0 checkpoint exists, has been evaluated against the frozen harness, and results are logged to the lineage tracker and W&B. Any pipeline bugs surface and get fixed here, on the cheapest variant, before the real ablation begins.

### Phase 4 — V1–V3 Ablation Runs
**Objective:** run the three remaining variants, each with exactly one additional curation decision applied, using the now-proven pipeline from Phase 3.
**Checkpoint:** all four variants (V0–V3) trained, checkpointed, and evaluated against the identical frozen harness; all four rows present in the lineage tracker with complete metrics.

### Phase 5 — Analysis and Write-Up
**Objective:** the actual deliverable — a written comparison of V0 through V3 across all evaluation metrics, with explicit discussion of which curation decision mattered, which didn't, and the classifier's popularity-vs-quality caveat.
**Checkpoint:** a report exists, stating a clear, evidence-backed conclusion for each of the three isolated questions (does filtering help, does dedup help, does the replay buffer measurably preserve general capability), plus documented limitations.

---

## 13. Stretch Goals (post-core, optional)

- Repeat the ablation on a second base model (e.g., Llama-3.2-1B) to test whether conclusions generalize
- Add curriculum ordering (easy-to-hard) as a fifth variant, testing sequencing rather than inclusion/exclusion
- A self-distillation round: use the best-performing checkpoint to generate more Q&A, filter through the quality classifier, and test whether another training round helps or plateaus
- Publish the curated dataset and ablation write-up publicly
- A minimal model-registry gate: block "promotion" of a new checkpoint unless it beats the previous best on the frozen eval set

---

## 14. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Eval set contamination | Carve out held-out set first, before any other stage touches the data; enforce by construction |
| Weak labels reflect popularity, not quality | State explicitly as a limitation in the Phase 5 write-up; do not oversell the classifier as a "correctness" detector |
| Free-tier compute session limits interrupt training | Checkpoint/resume built in from Phase 0 |
| Silent loss-masking bug invalidates SFT | Explicit verification step confirming loss is computed only on response tokens |
| Replay buffer ratio chosen arbitrarily | Treat as a tunable; briefly justify the chosen value in the write-up rather than presenting it as self-evidently correct |

### Open Items / Flagged Gaps

**Known gap:** Phase 0's parser retains only the accepted answer per question; every non-accepted answer is discarded at ingestion. Phase 4's DPO stage requires a lower-scored "rejected" answer per question to construct preference pairs — this data does not currently exist anywhere in the pipeline. Phase 4 will need a targeted re-parse of the dump (pulling non-accepted answers for questions already in the working pool) before DPO pair construction can begin. Not resolved as of Phase 1.

---

## 15. Definition of Done

- All nine pipeline stages implemented, idempotent, and independently re-runnable
- All four ablation variants (V0–V3) trained and evaluated against the single frozen held-out set
- A written report presenting quantified conclusions for each isolated curation decision
- Full reproducibility: any variant can be re-trained from its logged config and produce comparable results
- Lineage tracker and W&B logs complete for all runs
