# Alloy — Technical Glossary

Terms introduced while scoping and specifying Alloy, explained for someone comfortable with Python/SQL/general backend development but newer to ML training internals. Organized by category.

---

## Fine-tuning & Training Techniques

**LoRA (Low-Rank Adaptation)** — instead of updating all of a model's billions of parameters during fine-tuning, LoRA freezes the original weights and injects small, trainable "adapter" matrices alongside them. You train a tiny fraction of the parameters instead of the whole model — dramatically cheaper in memory and compute, while capturing most of the benefit of full fine-tuning.

**QLoRA** — LoRA combined with 4-bit quantization of the frozen base model: the base weights are compressed to 4-bit precision to save memory, while the small LoRA adapters are still trained normally. This is what makes fine-tuning a 1.5B+ parameter model feasible on a single consumer GPU.

**Continued pretraining** (also *continual pretraining* / CPT, or *domain-adaptive pretraining* / DAPT) — taking a model that's already been trained on general text and training it further on a narrower domain corpus, so it absorbs domain-specific vocabulary and knowledge before you fine-tune it for a specific task.

**Catastrophic forgetting** — when a model, while learning something new, loses previously-learned general knowledge or capability. The classic failure mode of naive continued pretraining or fine-tuning: you improve at your narrow domain but get measurably worse at everything else.

**Replay buffer** (in this context) — mixing in a slice of general-purpose text alongside domain-specific training data, specifically to counteract catastrophic forgetting. The model keeps "seeing" general language even while mainly learning the new domain.

**SFT (Supervised Fine-Tuning)** — the standard fine-tuning step where a model is trained on example input/output pairs (here: question → answer) with an ordinary supervised loss.

**Loss masking** — during SFT, deliberately excluding certain tokens from the loss calculation — here, the prompt/question tokens — so the model is only ever penalized for getting the *answer* wrong, not for "predicting" the question it was already given.

**DPO (Direct Preference Optimization)** — a method for teaching a model to prefer better outputs over worse ones, using pairs of (chosen response, rejected response) for the same input. Achieves a similar effect to full RLHF but without needing a separate reward model or the expensive trial-and-error rollout loop reinforcement learning requires.

**RLHF (Reinforcement Learning from Human Feedback)** — the older, more expensive paradigm DPO is a lighter-weight alternative to: train a separate reward model on human preference data, then use that reward model to guide the target model's behavior via reinforcement learning.

**PPO (Proximal Policy Optimization)** — the reinforcement learning algorithm classically used inside RLHF to update a model based on reward-model feedback. Mentioned mainly as a contrast point — Alloy uses DPO specifically to avoid needing PPO's rollout machinery.

**GRPO (Group Relative Policy Optimization)** — a newer RL variant used in some reasoning-model training pipelines, which compares a group of sampled outputs against each other instead of needing a separate value/reward model. Mentioned for context on how modern training pipelines are structured — not used in Alloy.

**Base model vs. Instruct model** — a "base" model is trained only to predict the next token on general text. An "instruct" model has been further trained (via SFT/RLHF/DPO) to follow instructions and hold a conversation. Fine-tuning projects like this one almost always start from an instruct-tuned checkpoint, since it's already usable out of the box.

**Checkpoint (ML sense)** — a saved snapshot of a model's weights at a particular point in training. Worth not confusing with the *project-phase* "checkpoint" used elsewhere in the spec — same word, two different meanings in this document.

---

## Data & Curation

**Ablation study** — an experiment where you start from a baseline and change exactly one variable at a time, measuring the effect of each change in isolation. The core methodology of this entire project.

**Weak labeling / weak supervision** — generating training labels from an imperfect, indirect signal (here: community upvotes and accepted-answer flags) instead of careful human annotation. Cheap and scalable, but noisier than hand-labeled data — the labels are a proxy for what you actually want to measure, not a direct measurement of it.

**Held-out set / eval set** — data deliberately set aside and never used in training, so it can be used afterward to measure performance on genuinely unseen examples.

**Near-duplicate detection / deduplication (dedup)** — identifying entries in a dataset that are effectively the same (or trivially reworded versions of each other) and removing all but one, so the model doesn't over-learn from repeated content.

**Embedding** — a numerical vector representation of a piece of text, produced by a model, such that texts with similar meaning end up with similar vectors. Used here both to detect near-duplicates (similar vectors = similar text) and as input features for the quality classifier.

**MinHash / LSH (Locality-Sensitive Hashing)** — a fast, approximate technique for finding near-duplicate text at scale without comparing every pair directly. It hashes documents so that similar documents are likely to land in the same hash "bucket," making duplicate detection far cheaper than brute-force comparison.

---

## Evaluation

**Perplexity** — a standard measure of how well a language model predicts a given piece of text — roughly, how "surprised" the model is by the actual next word at each step. Lower perplexity on the held-out set means the model fits that domain's language better.

**LLM-as-judge** — using a (typically stronger) language model to compare or score outputs from the model being evaluated, as a stand-in for human judgment. Used here to compare answer quality between ablation variants.

**MMLU (Massive Multitask Language Understanding)** — a widely-used benchmark testing a model's general knowledge across many unrelated academic and professional subjects. Used here in a small subset form, purely as a general-capability check to catch catastrophic forgetting — not as a headline result.

---

## Infrastructure & Tooling

**Unsloth** — an open-source library that makes LoRA/QLoRA fine-tuning, continued pretraining, and DPO significantly faster and more memory-efficient than standard Hugging Face tooling, largely through custom low-level kernels. The specific reason this project is feasible on free-tier GPUs at all.

**Weights & Biases (W&B)** — a hosted experiment-tracking tool. Every training run's config, metrics, and loss curves get logged automatically, so runs can be compared later without manually keeping a spreadsheet.

**YAML** — a human-readable text format commonly used for configuration files. In this project, every experiment run is defined by one YAML file rather than hardcoded parameters, so runs are easy to compare and reproduce.

**Idempotent (pipeline stage)** — a stage that produces the same result no matter how many times it's run with the same input. Critical for a pipeline where you'll frequently need to re-run just one stage after changing an upstream decision, without side effects piling up.

**Data lineage** — a record of exactly which inputs, code version, and configuration produced a given output. Here: tracking which config produced which curated dataset version produced which trained checkpoint, so every result is traceable back to the decisions that produced it.

---

## Hardware & Compute

**VRAM** — the memory built into a GPU itself (as opposed to your computer's regular RAM), where model weights and activations need to fit during training or inference. The single biggest constraint on what size model you can train or run locally.

**Tensor cores** — specialized circuitry on newer NVIDIA GPUs (RTX 20-series and later, but notably *not* the GTX 16-series, including the 1650) that dramatically accelerate the specific matrix-multiplication operations neural networks rely on. Their absence is why the GTX 1650 works fine for development but is slow for anything resembling real training.

**CUDA compute capability** — a version number NVIDIA assigns to each GPU generation, indicating which low-level GPU features and optimizations it supports. Training libraries often specify a minimum compute capability their optimized code paths require.
