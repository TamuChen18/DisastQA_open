# DisastQA GitHub Repository - Final Structure

## 📋 Data Pipeline Summary

Based on your description, the data pipeline is as follows:

1. **Raw Data** → `benchmark/corpus/ordered_corpus.json` (contains 5740 user_queries with score=3)
2. **Data Preparation** → `benchmark/MCQ/data_prepare.py` + `generate_mcq_set.py` 
3. **Human Annotation** → `DATA/DATA/annotation_mcq/ground_truth_MCQ_correctly_balanced.json` (2000 MCQs)
4. **Final MCQ** → `DATA/final_mcq/base_2000.json`, `golden_2000.json`, `mix_2000.json`
5. **OE Generation** → Select 1000 from remaining 3740 → `DATA/final_OE/base_oe.json`, `golden_oe.json`, `mix_oe.json`
6. **Model Evaluation** → `DATA/local_MCQ/` and `DATA/local_OE/` (results for each model)
7. **Analysis Results** → Various JSON and MD report files

## 📁 Suggested GitHub Repository Structure

```
DisastQA/
├── README.md                           # Main project documentation
├── LICENSE                             # MIT License
├── .gitignore                          # Git ignore file
├── requirements.txt                    # Python dependencies
├── 
├── 📁 data/                            # Dataset directory
│   ├── final_mcq/                     # Final MCQ dataset
│   │   ├── base_2000.json             # Base test set
│   │   ├── golden_2000.json           # Golden test set
│   │   └── mix_2000.json              # Mixed test set
│   │
│   ├── final_OE/                      # Final OE dataset
│   │   ├── base_oe.json               # Base OE test set
│   │   ├── golden_oe.json             # Golden OE test set
│   │   ├── mix_oe.json                # Mixed OE test set
│   │   ├── base_oe_with_difficulty.json
│   │   ├── golden_oe_with_difficulty.json
│   │   └── mix_oe_with_difficulty.json
│   │
│   ├── corpus/                        # Corpus
│   │   └── ordered_corpus.json        # Copy from benchmark/corpus/
│   │
│   ├── test_queries/                  # Original test queries
│   │   ├── QA_biological.json         # Copy from benchmark/test_query/
│   │   ├── FactCheck_chemical.json
│   │   └── [Other domain query files]
│   │
│   ├── annotation_mcq/                # Human-annotated MCQs
│   │   └── ground_truth_MCQ_correctly_balanced.json
│   │
│   └── sample_data/                   # Sample data
│       └── [Small-scale sample files]
│
├── 📁 code/                            # Code directory
│   ├── data_preparation/              # Data preparation scripts
│   │   ├── data_prepare.py            # Copy from benchmark/MCQ/
│   │   ├── generate_mcq_set.py        # Copy from benchmark/MCQ/
│   │   ├── generate_oe_set.py         # Copy from benchmark/OE/
│   │   └── generate_oe_from_mcq.py    # Copy from benchmark/OE/
│   │
│   ├── evaluation/                     # Evaluation scripts
│   │   ├── mcq_evaluation/            # MCQ evaluation
│   │   │   ├── local_evaluation.py    # Open-source model evaluation
│   │   │   ├── evaluate_closemodel.py # Closed-source model evaluation
│   │   │   └── analyze_MCQ_by_category.py
│   │   │
│   │   └── oe_evaluation/             # OE evaluation
│   │       ├── local_evaluation.py    # Open-source model evaluation
│   │       ├── local_evaluation_with_difficulty.py
│   │       └── evaluate_closemodel_oe.py # Closed-source model evaluation
│   │
│   └── analysis/                       # Analysis scripts
│       ├── summarize_MCQ_results.py
│       ├── calculate_golden_ranking.py
│       └── [Other analysis scripts]
│
├── 📁 results/                         # Experimental results directory
│   ├── mcq_results/                   # MCQ evaluation results
│   │   ├── MCQ_category_analysis_results.json
│   │   ├── MCQ_summary_by_intent_event.json
│   │   ├── MCQ_summary_by_intent_event.csv
│   │   └── golden_mcq_ranking.json
│   │
│   ├── oe_results/                    # OE evaluation results
│   │   ├── [OE-related result files]
│   │   └── [BLEU, ROUGE and other metric results]
│   │
│   ├── performance_reports/           # Performance reports
│   │   ├── detailed_mcq_performance_data.json
│   │   ├── detailed_mcq_performance_report.md
│   │   ├── mcq_performance_data.json
│   │   ├── mcq_performance_report.md
│   │   ├── oe_performance_data.json
│   │   ├── oe_performance_report.md
│   │   ├── golden_oe_analysis_data.json
│   │   ├── golden_oe_analysis_report.md
│   │   ├── oe_difficulty_performance_data.json
│   │   └── oe_difficulty_performance_report.md
│   │
│   └── statistics/                    # Statistical data
│       ├── difficulty_coverage_table.csv
│       └── keypoint_group_stats.csv
│
├── 📁 model_results/                   # Model evaluation results
│   ├── local_mcq/                     # Each model's MCQ results
│   │   ├── gpt-4o/
│   │   ├── llama-3-8b/
│   │   ├── phi-2/
│   │   └── [Other models]/
│   │
│   └── local_oe/                      # Each model's OE results
│       ├── gpt-4o/
│       ├── llama-3-8b/
│       ├── phi-2/
│       └── [Other models]/
│
└── 📁 docs/                            # Documentation directory
    ├── COMPLETE_PROJECT_OVERVIEW.md
    ├── methodology.md                 # Methodology description
    ├── experimental_setup.md          # Experimental setup
    └── results_interpretation.md      # Results interpretation
```

## 🎯 Core File List

### Files to Include

#### 1. Dataset Files
- `data/final_mcq/*.json` - Final MCQ test sets
- `data/final_OE/*.json` - Final OE test sets
- `data/corpus/ordered_corpus.json` - Corpus
- `data/test_queries/*.json` - Original test queries
- `data/annotation_mcq/ground_truth_MCQ_correctly_balanced.json` - Human-annotated MCQs

#### 2. Data Generation Scripts
- `code/data_preparation/data_prepare.py` - Data preparation script
- `code/data_preparation/generate_mcq_set.py` - MCQ generation script
- `code/data_preparation/generate_oe_set.py` - OE generation script
- `code/data_preparation/generate_oe_from_mcq.py` - OE generation from MCQ script

#### 3. Evaluation Scripts
- `code/evaluation/mcq_evaluation/local_evaluation.py` - Open-source model MCQ evaluation
- `code/evaluation/mcq_evaluation/evaluate_closemodel.py` - Closed-source model MCQ evaluation
- `code/evaluation/oe_evaluation/local_evaluation.py` - Open-source model OE evaluation
- `code/evaluation/oe_evaluation/local_evaluation_with_difficulty.py` - OE evaluation with difficulty
- `code/evaluation/oe_evaluation/evaluate_closemodel_oe.py` - Closed-source model OE evaluation

#### 4. Analysis Scripts
- `code/analysis/summarize_MCQ_results.py` - MCQ results summarization
- `code/analysis/calculate_golden_ranking.py` - Golden ranking calculation
- `code/analysis/analyze_MCQ_by_category.py` - Category-wise analysis

#### 5. Analysis Results
- `results/mcq_results/*.json` - MCQ analysis results
- `results/oe_results/*.json` - OE analysis results
- `results/performance_reports/*.md` - Performance reports
- `results/statistics/*.csv` - Statistical data

#### 6. Model Evaluation Results
- `model_results/local_mcq/*/` - Each model's MCQ results
- `model_results/local_oe/*/` - Each model's OE results

#### 7. Documentation
- `README.md` - Main project description
- `docs/COMPLETE_PROJECT_OVERVIEW.md` - Complete project overview
- All `.md` documentation files

### Files NOT to Include

#### 1. Large Model Files
- All model weight files in `models/` directory
- Pre-built index files

#### 2. Temporary Files
- `__pycache__/` directories
- Temporary result files
- Log files

#### 3. System Files
- `ngrok-test/` related files
- Archive files

## 📝 Suggested README.md Structure

```markdown
# DisastQA: Large-Scale RAG Evaluation Framework

## Project Overview
This project builds a large-scale, multi-dimensional RAG evaluation framework for systematically assessing the performance of different retrieval methods, generation models, and configuration combinations on question-answering tasks.

## Key Features
- Large-scale experiments: 1.8M+ question processing, 312 experimental configurations
- Multi-dimensional evaluation: 8 models × 6 retrieval methods × 39 configuration combinations
- Dual task support: MCQ (Multiple Choice Questions) + OE (Open-ended Questions)
- Human annotation: 2000 MCQs + 1000 OEs of high-quality human-annotated data

## Dataset
- MCQ test sets: `data/final_mcq/` (base, golden, mix, each with 2000)
- OE test sets: `data/final_OE/` (base, golden, mix, each with 1000)
- Corpus: `data/corpus/ordered_corpus.json` (239K documents)
- Original test queries: `data/test_queries/` (categorized by domain)

## Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Run evaluation: `python code/evaluation/mcq_evaluation/local_evaluation.py`
3. View results: `results/performance_reports/`

## Experimental Results
Detailed results please refer to the report files in `results/` directory.

## Citation
If you use this project, please cite our paper...
```

## 🔧 Create .gitignore File

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/

# Large model files
models/
*.bin
*.safetensors
*.pt
*.pth

# Pre-built index files
indexes/
*.pkl
*.index
embeddings.npy

# Temporary files
*.tmp
*.log
*.cache

# System files
.DS_Store
Thumbs.db

# Archive files
*.zip
*.tar.gz
*.tgz

# Large data files
# data/corpus/ordered_corpus.json

# Large files in experimental results directories
model_results/*/detailed_results/

# Log files
*.log

# Database files
*.db
*.sqlite

# IDE files
.vscode/
.idea/

# Jupyter Notebook checkpoints
.ipynb_checkpoints/

# Environment variable files
.env

# Other unnecessary files
ngrok-test/
ngrok-*
```

## 📊 Repository Size Estimation

- Core code: ~50MB
- Dataset: ~200MB (compressed)
- Analysis results: ~100MB
- Model results: ~200MB (compressed)
- Documentation: ~10MB
- **Total**: ~560MB (within GitHub limits)

This structure more accurately reflects the actual content of your DisastQA project.