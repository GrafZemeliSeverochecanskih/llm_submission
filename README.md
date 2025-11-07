# Cross-Lingual Emotion Classification and Detection Using Large Language Models

**Author:** Simion Polivencu (S5480183)  
**Date:** October 2025  
**Repository:** [GrafZemeliSeverochecanskih/llm_submission](https://github.com/GrafZemeliSeverochecanskih/llm_submission)

---

## 📘 Overview

This repository contains the notebooks and results for the project *Cross-Lingual Emotion Classification and Detection Using Large Language Models*.  
The study investigates whether multilingual emotion classification can be effectively handled by lightweight, parameter-efficient language models using **Low-Rank Adaptation (LoRA)**, and how **prompt engineering efficiency** scales with model size.

The entire workflow — including data preprocessing, LoRA fine-tuning, and prompt-based evaluation — is implemented in **Jupyter Notebooks (`.ipynb`)**.

---

## 🎯 Research Goals

1. **Evaluate LoRA-based fine-tuning** on multilingual emotion classification to determine whether small or medium models can handle complex multilingual semantic tasks efficiently.
2. **Assess the role of prompt engineering** in guiding large instruction-tuned models without additional training, focusing on zero-shot, few-shot, and instruction-based prompts.
3. **Compare scaling behavior** across models with different parameter sizes and training alignments (base, instruction-tuned, chat-aligned).

---

## 🧠 Methodology

### Dataset

The experiments use the **XED Multilingual Emotion Dataset** ([Helsinki-NLP/XED](https://github.com/Helsinki-NLP/XED)), which contains movie subtitle phrases annotated with numeric emotion identifiers (1–8) based on **Plutchik’s eight emotions**:
`joy, sadness, anger, fear, surprise, anticipation, trust, disgust`.

Each data point:  
(text, [emotion_ids])

where `emotion_ids` can be single or multi-label.  
Data was preprocessed to remove duplicates, correct corrupted samples, and generate both monolingual (EN, FR, ES, FI) and multilingual combined splits.

---

### LoRA Fine-Tuning

LoRA adapters were integrated into attention and feed-forward layers using the **Hugging Face PEFT library**.  
Training was performed on smaller and larger model cohorts:

| Cohort | Model Examples | Size | Tuning Strategy | Purpose |
|---------|----------------|------|------------------|----------|
| Cohort 1 | `flan-t5-base`, `gpt2-medium`, `TurkuNLP/gpt3-finnish-medium`, etc. | 250M–1.6B | Grid search over (r, α, dropout) | Hyperparameter optimization |
| Cohort 2 | `EleutherAI/gpt-neo-1.3B`, `Ahma-3B`, `Qwen2.5-3B-Instruct`, `Falcon3-3B` | 1.3B–3.6B | Fixed config (r=32, α=16, dropout=0.1) | Large-scale evaluation |

Fine-tuning was run for 2–5 epochs with mixed precision and frozen base weights.  
Evaluation metrics were computed on 300 test samples per language.

---

### Prompt Engineering

Prompt-based evaluations were performed without fine-tuning to measure the inherent **instruction-following** capability of models.  
Three prompting strategies were compared:

- **Zero-shot:** Only the task instruction and input text.  
- **Few-shot:** Adds a few labeled examples before the query.  
- **Instruction-based:** Includes detailed label mappings and explicit numeric output formatting.

Models evaluated:

| Model | Parameters | Architecture | Type |
|--------|-------------|---------------|------|
| `google/flan-t5-base` | 250M | Encoder–Decoder | Instruction-tuned |
| `TinyLlama-1.1B-Chat` | 1.1B | Decoder-only | Chat-aligned (DPO) |
| `Mistral-7B-Instruct-v0.2` | 7B | Decoder-only | Instruction-tuned / RLHF |

---

## 📊 Evaluation Metrics

Two metrics were used for all experiments:

- **Macro F1-Score** — balances precision and recall across all emotion categories, mitigating class imbalance.
- **Jaccard Index** — measures the overlap between predicted and true multi-label emotion sets.

These metrics provide a balanced assessment of multi-label classification accuracy and semantic consistency.

---

## 📈 Key Results

### LoRA Fine-Tuning
| Language | Initial Macro F1 | Post-LoRA Macro F1 | Initial Jaccard | Post-LoRA Jaccard |
|-----------|------------------|--------------------|-----------------|-------------------|
| English | 0.03 | 0.45 | 0.01 | 0.44 |
| Finnish | 0.06 | 0.30 | 0.04 | 0.28 |
| French | 0.03 | 0.31 | 0.02 | 0.27 |
| Spanish | 0.01 | 0.33 | 0.01 | 0.30 |
| Multilingual | 0.04 | 0.36 | 0.02 | 0.34 |

LoRA adaptation led to substantial performance improvements across all languages with <1% additional trainable parameters, confirming its effectiveness for multilingual semantic tasks.

---

### Prompt Engineering (Flan-T5 Example)
| Strategy | English Macro F1 | French Macro F1 | Multilingual Macro F1 |
|-----------|------------------|-----------------|------------------------|
| Zero-shot | 0.00 | 0.00 | 0.00 |
| Few-shot | 0.00 | 0.00 | 0.00 |
| Instruction-based | 0.10 | 0.06 | 0.05 |

Instruction-based prompting significantly improved prediction consistency, particularly in English and French, showing that explicit label mapping and structured responses enhance classification behavior.

---

### Model Comparison Summary
- **Flan-T5-base (LoRA)**: Strong cross-lingual generalization after adaptation.  
- **TinyLlama-1.1B-Chat (Prompting)**: Gains from structured prompts despite small size.  
- **Mistral-7B-Instruct (Prompting)**: Best zero-shot and instruction-based results; larger context window helps cross-language inference.

---

## 💡 Discussion

- LoRA enables efficient fine-tuning with minimal resource overhead while retaining strong multilingual generalization.
- Prompt engineering alone yields limited results unless task definitions and output constraints are explicitly provided.
- Larger models demonstrate better prompt-following behavior, but parameter-efficient fine-tuning remains essential for task specialization.

---

## 🧾 Conclusion

This project confirms that **hybrid adaptation strategies**—combining LoRA fine-tuning for task alignment with optimized prompting for flexible inference—offer an effective path for **scalable, multilingual emotion classification**.  
Future work may explore **automated prompt optimization**, **larger balanced datasets**, and **PEFT variants** like QLoRA or AdapterFusion.

---
