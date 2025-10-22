# DisastQA: Large-Scale RAG Evaluation Framework

## 📋 Project Overview

This project builds a large-scale, multi-dimensional RAG (Retrieval-Augmented Generation) evaluation framework for systematically assessing the performance of different retrieval methods, generation models, and configuration combinations on question-answering tasks. This is one of the most comprehensive RAG system evaluation studies to date.

### 🎯 Key Features
- **Large-scale Experiments**: 1.8M+ question processing, 312 experimental configurations
- **Multi-dimensional Evaluation**: 8 models × 6 retrieval methods × 39 configuration combinations
- **Dual Task Support**: MCQ (Multiple Choice Questions) + OE (Open-ended Questions)
- **Human Annotation**: High-quality human-annotated data with 2000 MCQs + 1000 OEs
- **Complete Pipeline**: End-to-end solution from data construction to result analysis

## 🏗️ System Architecture

### RAG Pipeline Architecture
```
User Query → Retrieval Layer → Re-ranking Layer → Generation Layer → Final Answer
    ↓            ↓              ↓                ↓              ↓
  Question   BM25/Vector   CrossEncoder        LLM         Answer
```

### Technology Stack
- **Retrieval**: BM25, Vector Retrieval, Hybrid Retrieval, Elasticsearch
- **Re-ranking**: CrossEncoder (ms-marco-MiniLM-L-6-v2)
- **Generation**: 7 local models + 1 API model
- **Evaluation**: Accuracy, ROUGE, BLEU, BERTScore
- **Engineering**: Parallel Processing, Batch Processing Optimization

## 📊 Experimental Scale

### Dataset Scale
- **Corpus**: 239,704 documents
- **Test Questions**: 5,740 high-quality questions
- **Annotated Data**: Complete relevance annotations for 7,998 queries
- **Domain Coverage**: Biology, Chemistry, Environment, Geology, Meteorology, Society, Technology, etc.

### Experimental Configuration
- **Number of Models**: 8 (7 local + 1 API)
- **Retrieval Methods**: 6 types (keyword_only, vector_only, hybrid_only, keyword_rerank, vector_rerank, hybrid_rerank)
- **Configuration Combinations**: 39 types (different retrieval quantities and re-ranking parameters)
- **Total Experiments**: 312
- **Total Question Processing**: 1,791,360 questions

## 📁 Project Structure

```
DisastQA/
├── data/                    # Dataset directory
│   ├── final_mcq/          # Final MCQ dataset
│   ├── final_OE/           # Final OE dataset
│   ├── corpus/             # Corpus
│   ├── test_queries/       # Original test queries
│   └── annotation_mcq/     # Human-annotated data
├── code/                   # Code directory
│   ├── data_preparation/   # Data preparation scripts
│   ├── EXPERIENCE/         # Evaluation scripts
│   └── analysis/           # Analysis scripts
├── results/                # Experimental results
├── model_results/          # Model evaluation results
└── docs/                   # Documentation
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Clone the project
git clone [your-repo-url]
cd DisastQA

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation
```bash
# Run data preparation scripts
python code/data_preparation/data_prepare.py
python code/data_preparation/generate_mcq_set.py
```

### 3. Run Evaluation
```bash
# Run MCQ evaluation
python code/evaluation/mcq_evaluation/local_evaluation.py

# Run OE evaluation
python code/evaluation/oe_evaluation/local_evaluation.py
```

### 4. View Results
```bash
# View experimental results
ls results/performance_reports/
cat results/performance_reports/mcq_performance_report.md
```

## 📈 Main Experimental Results

### Retrieval Method Performance Ranking
| Rank | Method | MRR@1 | MAP@10 | Features |
|------|--------|-------|--------|----------|
| 1 | Elasticsearch | 0.550 | 0.353 | Best overall performance |
| 2 | BM25+ | 0.538 | 0.343 | Best traditional algorithm |
| 3 | BM25 | 0.510 | 0.321 | Standard BM25 |
| 4 | TF-IDF | 0.448 | 0.293 | Basic vector method |

### Model Performance Ranking (MCQ Task)
| Rank | Model | Size | Accuracy | Features |
|------|-------|------|----------|----------|
| 1 | gpt-4o | Large | 95.5% | Best commercial model |
| 2 | qwen-3-8b | 8B | 94.9% | Best open-source model |
| 3 | phi-2 | 2.7B | 94.0% | Best small model |
| 4 | llama-3-8b | 8B | 94.0% | Stable performance |

### Configuration Optimization Findings
- **Best Retrieval Method**: BM25 + Vector retrieval hybrid
- **Best Retrieval Quantity**: 25 documents → re-rank to 8
- **Performance Improvement**: 15-20% improvement over single methods
- **Cost-Effectiveness**: Local models outperform API models in cost-effectiveness

## 📊 Dataset Description

### MCQ Dataset
- **base_2000.json**: Base test set, 2000 questions
- **golden_2000.json**: Golden test set, 2000 questions
- **mix_2000.json**: Mixed test set, 2000 questions

### OE Dataset
- **base_oe.json**: Base OE test set, 1000 questions
- **golden_oe.json**: Golden OE test set, 1000 questions
- **mix_oe.json**: Mixed OE test set, 1000 questions

### Corpus
- **ordered_corpus.json**: Corpus with 239,704 documents

## 🔧 Technical Details

### Model Configuration Strategy
```python
# Small model configuration (1-2B)
{
    "batch_size": 32,
    "num_workers": 8,
    "rerank_batch_size": 256,
    "torch_dtype": torch.float16
}

# Medium model configuration (7-8B)
{
    "batch_size": 16,
    "num_workers": 16,
    "rerank_batch_size": 128,
    "torch_dtype": torch.float16
}
```

### Evaluation Metrics
- **MCQ Task**: Accuracy
- **OE Task**: ROUGE-L, BLEU-4, BERTScore-F1
- **Retrieval Task**: MRR@1, MAP@10, NDCG@10

## 📚 Related Documentation

- [Complete Project Overview](docs/COMPLETE_PROJECT_OVERVIEW.md)
- [Methodology](docs/methodology.md)
- [Experimental Setup](docs/experimental_setup.md)
- [Results Interpretation](docs/results_interpretation.md)

## 🤝 Contributing

### How to Contribute
1. Fork this project
2. Create a feature branch
3. Submit code changes
4. Create a Pull Request

### Issue Reporting
- Use GitHub Issues to report bugs
- Provide detailed error information and reproduction steps
- Include system environment information

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

- Project Maintainer: [Your Name]
- Email: [your.email@example.com]
- Project URL: [GitHub Repository URL]

## 🙏 Acknowledgments

Thanks to all researchers and developers who contributed to this project.

---

*Last updated: December 2024*