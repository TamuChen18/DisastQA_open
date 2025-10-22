# 🧠 DisastQA: Datasets and Evaluation for Disaster-Domain Question Answering

**DisastQA** is the first large-scale benchmark for disaster-response question answering (QA).
It explicitly models retrieval uncertainty and factual completeness—two defining characteristics of real disaster scenarios—to evaluate how well LLMs reason and respond under incomplete, noisy, or conflicting information.

DisastQA provides:
- 3,000 rigorously verified questions (2,000 MCQ + 1,000 OE)
- 8 disaster types across meteorological, geophysical, hydrological, climatological, biological, technological, extraterrestrial, and conflict-induced events
- Human–LLM collaborative construction pipeline with expert validation
- Novel keypoint-based evaluation for factual completeness
- Retrieval-aware settings: Base, Mix, and Golden to separate model knowledge from evidence use

---

## 🌍 Motivation

Accurate QA during disasters is critical for decision-making and crisis management, yet existing benchmarks (e.g., SQuAD, MMLU-Pro) target general or professional domains and fail to capture disaster-specific uncertainty.
Real-world disaster information is fragmented, noisy, and multi-fact—models must synthesize multiple atomic facts (e.g., hazard, location, casualty, response).

DisastQA bridges this gap by providing a retrieval-aware, factual, and human-validated benchmark for evaluating LLM reliability in high-stakes scenarios.

---

## 📚 Key Features and Contributions

- Retrieval-Aware Design: Evaluates models under Base (no context), Mix (noisy retrieval), and Golden (oracle evidence) settings.
- Human–LLM Pipeline: Combines LLM scalability with expert validation and rewriting (~40% human-revised).
- Keypoint-Based Evaluation: A novel factual-completeness metric measuring how many atomic facts a model correctly reproduces.
- Dual QA Tracks:
  - MCQ: Tests discriminative reasoning and factual discrimination.
  - OE: Tests factual completeness, keypoint recall, and reasoning depth.
- Comprehensive Evaluation: 18 LLMs (0.6B–8B + APIs) benchmarked under consistent retrieval contexts.

---

## 🧩 Repository Structure

```
DisastQA/
├── benchmark/
│   ├── MCQ/
│   │   ├── data_prepare.py
│   │   └── generate_mcq_set.py
│   └── OE/
│       ├── generate_oe_set.py
│       └── generate_oe_from_mcq.py
│
├── DATA/
│   ├── final_mcq/                   # base_2000.json, mix_2000.json, golden_2000.json
│   ├── final_OE/                    # base_oe.json, ... , base_oe_with_difficulty.json
│   ├── local_MCQ/                   # model-specific MCQ results
│   ├── local_OE/                    # model-specific OE results
│   ├── MCQ_evaluation/              # evaluation scripts (local/closed)
│   ├── OE_evaluation/               # evaluation scripts (local/difficulty)
│   └── DATA/                        # annotation/intermediate artifacts
│
├── assets/
│   ├── pipeline.png
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## ⚙️ Data Pipeline Summary

1. Extract (query, passage) pairs with relevance score = 3 from DisastIR corpus.
2. Use an LLM to rewrite queries → QA-style questions (MCQ/OE).
3. Human annotators rewrite, validate, and construct distractors or reference answers.
4. Verify every item through multi-annotator solve-checking and consensus resolution.
5. Generate difficulty levels via keypoint count (OE).
6. Evaluate 18 models under Base, Mix, and Golden settings.


---

## 🧮 DisastQA Construction Pipeline

<p align="center">
  <img src="assets/pipeline.png" alt="DisastQA Construction Pipeline" width="85%">
</p>

*Figure: Overview of the Human–LLM collaborative pipeline for DisastQA construction and model evaluation. The benchmark is built via a Human–LLM collaboration workflow, covering query rewriting, human validation, and keypoint-based evaluation.*

---


## 🚀 Quick Start

### 1️⃣ Setup
```bash
pip install -r requirements.txt
```

### 2️⃣ Run MCQ Evaluation
```bash
python DATA/MCQ_evaluation/local_evaluation.py \
  --mcq_path DATA/final_mcq/base_2000.json \
  --model_name <your_local_model>
```

### 3️⃣ Run OE Evaluation (Keypoint-Aware)
```bash
python DATA/OE_evaluation/local_evaluation_with_difficulty.py \
  --oe_path DATA/final_OE/base_oe_with_difficulty.json \
  --model_name <your_local_model>
```

### 4️⃣ Regenerate Benchmark (Optional)
- MCQ: `benchmark/MCQ/data_prepare.py` → `benchmark/MCQ/generate_mcq_set.py`
- OE: `benchmark/OE/generate_oe_from_mcq.py` or `benchmark/OE/generate_oe_set.py`

⚠️ Keep all folder names and paths unchanged for compatibility.

---

## 📊 Metrics

| Task | Metric | Purpose |
| --- | --- | --- |
| MCQ | Accuracy | Correct answer discrimination |
| OE | Keypoint Coverage | Factual completeness |
| OE | ROUGE-L / BLEU-4 / BERTScore | Surface-level overlap (for reference) |

Keypoint coverage explicitly measures whether models recall all essential disaster facts—e.g., hazard type, impact region, casualties, and response measures—offering a more trustworthy assessment than traditional overlap metrics.

---

## 🧠 Highlighted Findings

- Retrieval quality strongly governs performance: Base < Mix < Golden across all models.
- DisastQA rankings diverge from general-domain benchmarks (Spearman’s ρ ≈ 0.2 vs. MMLU-Pro).
- GPT-4o achieves 95.4% factual coverage, while open models like Qwen-3-8B reach 99.6% MCQ accuracy under Golden retrieval.
- High factual density: OE answers contain on average 4.4 atomic keypoints, reflecting multi-fact reasoning complexity.
- Surface metrics (ROUGE/BERTScore) overestimate factuality—models often omit crucial details even when fluently written.

---

## 📈 Representative Results (Summary)

| Model | Params | MCQ (Golden) | OE Coverage (%) | Comment |
| --- | --- | --- | --- | --- |
| GPT-4o | — | 99.4 | 95.4 | Best factual completeness |
| Gemini-1.5 Pro | — | 98.7 | 95.1 | Balanced fluency & accuracy |
| Qwen-3-8B | 8B | 99.6 | 93.9 | Best open-source MCQ model |
| Llama-3-8B | 8B | 99.1 | 93.4 | Strong reasoning; factual gaps |
| Gemma-7B | 8.5B | 98.0 | 89.7 | Best ROUGE-L, lower factuality |

(Full results and difficulty breakdowns are available in the paper and Appendix.)

---

## 🧩 Insights and Impact

- Retrieval-aware evaluation is essential for safety-critical domains; general-domain performance is not predictive of real-world reliability.
- Human-in-the-loop data curation remains vital for factual precision—LLM-only generation introduces ambiguity and hallucination.
- Keypoint-based evaluation provides a generalizable method for assessing factual completeness across any domain.

DisastQA thus serves as a foundation for future work on:
- Multi-turn reasoning and dynamic retrieval
- Domain adaptation and continual learning in emergencies
- Robust factual evaluation for trustworthy AI

---

## 🪪 License

This project is released under the MIT License. See `LICENSE` for details.

---

## 📬 Citation

If you use DisastQA in your research, please cite:

```
@dataset{chen2025disastqa,
  title   = {DisastQA: A Benchmark for Disaster-Domain Question Answering},
  author  = {Chen, Zhitong and collaborators},
  year    = {2025},
  url     = {https://github.com/TamuChen18/DisastQA}
}
```

---

## 🙌 Acknowledgments

This benchmark was developed by researchers at Texas A&M University and collaborators in disaster resilience and information retrieval.
We thank all annotators and domain experts for their contributions.

---

## 🔗 Repository

- Maintainer: Zhitong Chen
- Repository: https://github.com/TamuChen18/DisastQA
