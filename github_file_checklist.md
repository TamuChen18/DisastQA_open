# GitHub Repository File Checklist for DisastQA

## ✅ Files to Include

### 📁 Root Directory Files
- [x] `README.md` - Main project documentation
- [x] `LICENSE` - MIT License
- [x] `.gitignore` - Git ignore file
- [x] `requirements.txt` - Python dependencies

### 📁 Dataset Files (data/)
- [ ] `data/final_mcq/base_2000.json` - Base MCQ test set
- [ ] `data/final_mcq/golden_2000.json` - Golden MCQ test set
- [ ] `data/final_mcq/mix_2000.json` - Mixed MCQ test set
- [ ] `data/final_OE/base_oe.json` - Base OE test set
- [ ] `data/final_OE/golden_oe.json` - Golden OE test set
- [ ] `data/final_OE/mix_oe.json` - Mixed OE test set
- [ ] `data/final_OE/base_oe_with_difficulty.json` - OE test set with difficulty
- [ ] `data/final_OE/golden_oe_with_difficulty.json` - Golden OE test set with difficulty
- [ ] `data/final_OE/mix_oe_with_difficulty.json` - Mixed OE test set with difficulty
- [ ] `data/corpus/ordered_corpus.json` - Corpus (copy from benchmark/corpus/)
- [ ] `data/test_queries/` - Original test queries (copy from benchmark/test_query/)
- [ ] `data/annotation_mcq/ground_truth_MCQ_correctly_balanced.json` - Human-annotated MCQs
- [ ] `data/sample_data/` - Sample data

### 📁 Code Files (code/)
#### Data Preparation Scripts (code/data_preparation/)
- [ ] `code/data_preparation/data_prepare.py` - Copy from benchmark/MCQ/
- [ ] `code/data_preparation/generate_mcq_set.py` - Copy from benchmark/MCQ/
- [ ] `code/data_preparation/generate_oe_set.py` - Copy from benchmark/OE/
- [ ] `code/data_preparation/generate_oe_from_mcq.py` - Copy from benchmark/OE/

#### Evaluation Scripts (code/evaluation/)
- [ ] `code/evaluation/mcq_evaluation/local_evaluation.py` - Copy from DATA/MCQ_evaluation/
- [ ] `code/evaluation/mcq_evaluation/evaluate_closemodel.py` - Copy from DATA/MCQ_evaluation/
- [ ] `code/evaluation/mcq_evaluation/analyze_MCQ_by_category.py` - Copy from DATA/MCQ_evaluation/
- [ ] `code/evaluation/oe_evaluation/local_evaluation.py` - Copy from DATA/OE_evaluation/
- [ ] `code/evaluation/oe_evaluation/local_evaluation_with_difficulty.py` - Copy from DATA/OE_evaluation/
- [ ] `code/evaluation/oe_evaluation/evaluate_closemodel_oe.py` - Copy from DATA/OE_evaluation/

#### Analysis Scripts (code/analysis/)
- [ ] `code/analysis/summarize_MCQ_results.py` - Copy from DATA/
- [ ] `code/analysis/calculate_golden_ranking.py` - Copy from DATA/
- [ ] `code/analysis/analyze_mmlu_pro_results.py` - Copy from root directory
- [ ] `code/analysis/convert_mmlu_pro_subjects.py` - Copy from root directory
- [ ] `code/analysis/calculate_coverage.py` - Copy from root directory
- [ ] `code/analysis/extract_keypoints.py` - Copy from root directory
- [ ] `code/analysis/generate_difficulty_table.py` - Copy from root directory

### 📁 Experimental Results (results/)
#### MCQ Results (results/mcq_results/)
- [ ] `results/mcq_results/MCQ_category_analysis_results.json` - Copy from DATA/
- [ ] `results/mcq_results/MCQ_summary_by_intent_event.json` - Copy from DATA/
- [ ] `results/mcq_results/MCQ_summary_by_intent_event.csv` - Copy from DATA/
- [ ] `results/mcq_results/golden_mcq_ranking.json` - Copy from DATA/

#### Performance Reports (results/performance_reports/)
- [ ] `results/performance_reports/detailed_mcq_performance_data.json` - Copy from root directory
- [ ] `results/performance_reports/detailed_mcq_performance_report.md` - Copy from root directory
- [ ] `results/performance_reports/mcq_performance_data.json` - Copy from root directory
- [ ] `results/performance_reports/mcq_performance_report.md` - Copy from root directory
- [ ] `results/performance_reports/oe_performance_data.json` - Copy from root directory
- [ ] `results/performance_reports/oe_performance_report.md` - Copy from root directory
- [ ] `results/performance_reports/golden_oe_analysis_data.json` - Copy from root directory
- [ ] `results/performance_reports/golden_oe_analysis_report.md` - Copy from root directory
- [ ] `results/performance_reports/oe_difficulty_performance_data.json` - Copy from root directory
- [ ] `results/performance_reports/oe_difficulty_performance_report.md` - Copy from root directory

#### Statistical Data (results/statistics/)
- [ ] `results/statistics/difficulty_coverage_table.csv` - Copy from root directory
- [ ] `results/statistics/keypoint_group_stats.csv` - Copy from root directory

### 📁 Model Evaluation Results (model_results/)
#### MCQ Results (model_results/local_mcq/)
- [ ] `model_results/local_mcq/` - Copy from DATA/local_MCQ/ (results for each model)

#### OE Results (model_results/local_oe/)
- [ ] `model_results/local_oe/` - Copy from DATA/local_OE/ (results for each model)

### 📁 Documentation (docs/)
- [ ] `docs/COMPLETE_PROJECT_OVERVIEW.md` - Copy from root directory

## ❌ Files NOT to Include

### Large Model Files
- [x] All model weight files in `models/` directory
- [x] Pre-built index files (`indexes/`, `*.pkl`, `*.index`, `embeddings.npy`)

### Temporary Files
- [x] `__pycache__/` directories
- [x] Temporary result files
- [x] Log files (`*.log`)

### System Files
- [x] `ngrok-test/` related files
- [x] Archive files (`*.zip`, `*.tar.gz`, `*.tgz`)

### Other Unnecessary Files
- [x] `rag_hprc/` directory (HPRC cluster-specific code)
- [x] `cognita/` directory (practical application system, not core to paper)
- [x] Large files in `benchmark/` directory (only copy needed scripts)

## 📊 Repository Size Estimation

- Core code: ~50MB
- Dataset: ~200MB (compressed)
- Analysis results: ~100MB
- Model results: ~200MB (compressed)
- Documentation: ~10MB
- **Total**: ~560MB (within GitHub limits)

## 🚀 Deployment Steps

1. **Create GitHub Repository**
2. **Copy Necessary Files** (according to the checklist above)
3. **Create Directory Structure** (according to suggested directory structure)
4. **Commit to GitHub**
5. **Create Release** (including dataset and analysis results)

## 📝 Notes

1. **Corpus File**: `data/corpus/ordered_corpus.json` might be large, need to check size
2. **Model Results**: Only include summary results, not detailed raw outputs
3. **API Keys**: Ensure no API keys or sensitive information are included
4. **Path Adjustment**: Need to adjust path references in code when copying files