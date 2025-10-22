# DisastQA: Datasets and Evaluation for Disaster-domain QA

## Overview
DisastQA provides finalized datasets and evaluation scripts for disaster-domain question answering. It supports two tasks:
- MCQ: Multiple-choice questions
- OE: Open-ended questions

This repository contains:
- `benchmark/`: scripts to generate MCQ/OE test sets from corpus/query/qrels
- `DATA/`: finalized datasets, human annotations, and evaluation results/scripts
- Root-level analysis scripts and reports

Folder names and paths are stable and must NOT be changed.

## Repository Structure
```
DisastQA/
├── benchmark/
│   ├── MCQ/
│   │   ├── data_prepare.py             # Build (query, passage) with score=3
│   │   └── generate_mcq_set.py         # Generate base/mix/golden MCQ sets
│   └── OE/
│       ├── generate_oe_set.py          # Generate OE from corpus (if used)
│       └── generate_oe_from_mcq.py     # Generate OE from MCQ leftovers
│
├── DATA/
│   ├── final_mcq/                      # Finalized MCQ test sets
│   │   ├── base_2000.json
│   │   ├── mix_2000.json
│   │   └── golden_2000.json
│   ├── final_OE/                       # Finalized OE test sets and reviews
│   │   ├── base_oe.json
│   │   ├── mix_oe.json
│   │   ├── golden_oe.json
│   │   ├── base_oe_with_difficulty.json
│   │   └── Human_review_question/ ...  # Manually reviewed subset (~200)
│   ├── DATA/                           # Intermediate/annotation materials
│   │   ├── annotation_mcq/
│   │   │   └── ground_truth_MCQ_correctly_balanced.json  # Human-verified 2000 MCQs
│   │   └── data_prepare/ ...           # score=3 selections per event type
│   ├── MCQ_evaluation/                 # MCQ evaluation scripts
│   │   ├── local_evaluation.py         # For local/open models
│   │   └── evaluate_closemodel.py      # For closed-source API models
│   ├── OE_evaluation/                  # OE evaluation scripts
│   │   ├── local_evaluation.py
│   │   └── local_evaluation_with_difficulty.py
│   ├── local_MCQ/                      # Model-specific MCQ results
│   ├── local_OE/                       # Model-specific OE results
│   └── MMLUE-PRO/                      # MMLU-PRO subset used in MCQ
│
├── requirements.txt
├── LICENSE
└── README.md
```

## Data Pipeline (Summary)
1. In `benchmark/MCQ/data_prepare.py`, build per-file mappings from `qrels` and `test_query`; extract pairs with score=3.
2. In `benchmark/MCQ/generate_mcq_set.py`, generate candidate MCQs; then human annotate/correct.
3. Human-balanced selection to 2000 MCQs → `DATA/DATA/annotation_mcq/ground_truth_MCQ_correctly_balanced.json`.
4. Final MCQ test sets written to `DATA/final_mcq/` as `base_2000.json`, `mix_2000.json`, `golden_2000.json`.
5. OE sets generated from remaining (5740−2000) balanced items → `DATA/final_OE/` as `base_oe.json`, etc. Difficulty variants consider keypoint counts.
6. Evaluation results for models are stored under `DATA/local_MCQ/` and `DATA/local_OE/`.

## Quick Start
### 1) Environment
```bash
pip install -r requirements.txt
```

### 2) MCQ Evaluation (local models)
```bash
python DATA/MCQ_evaluation/local_evaluation.py \
  --mcq_path DATA/final_mcq/base_2000.json \
  --model_name <your_local_model>
```

### 3) OE Evaluation (local models, difficulty-aware)
```bash
python DATA/OE_evaluation/local_evaluation_with_difficulty.py \
  --oe_path DATA/final_OE/base_oe_with_difficulty.json \
  --model_name <your_local_model>
```

### 4) Closed-source models
Use the corresponding `evaluate_closemodel*.py` in `DATA/MCQ_evaluation/` or `DATA/OE_evaluation/`.

### 5) Regenerate datasets (optional)
- MCQ: `benchmark/MCQ/data_prepare.py` → `benchmark/MCQ/generate_mcq_set.py`
- OE: `benchmark/OE/generate_oe_from_mcq.py` or `benchmark/OE/generate_oe_set.py`

Note: Do not change directory names or relative paths.

## Metrics
- MCQ: Accuracy
- OE: ROUGE-L, BLEU-4, BERTScore-F1

## Notes
- Large models, prebuilt indexes, and temporary files are excluded via `.gitignore`.
- `DATA/MMLUE-PRO` is included as plain data; embedded git metadata was removed.
- All documentation, comments, and code are in English.

## License
This project is licensed under the MIT License. See `LICENSE` for details.

## Contact
- Maintainer: DisastQA Team
- Repository: https://github.com/TamuChen18/DisastQA