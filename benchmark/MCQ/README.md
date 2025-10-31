cat > MCQ_README_full.txt <<'EOF'
# Multiple-Choice (MCQ) Question Generation System

This directory contains the Multiple-Choice Question (MCQ) Generation System, designed to create, evaluate, and benchmark large-scale multiple-choice datasets for RAG (Retrieval-Augmented Generation) evaluation and model comparison.

---

## 🔄 Differences from OE System

| Aspect | OE System | MCQ System |
|--------|------------|------------|
| **Question Type** | Open-ended questions with long answers | Multiple-choice questions (4 options) |
| **Answer Format** | 3–5 paragraph detailed answer | Single correct letter (A/B/C/D) |
| **Evaluation Focus** | Comprehensiveness and reasoning | Option quality, correctness, distractor plausibility |
| **Output Structure** | `open_ended.gpt40.content` | `multiple_choice.gpt40.content` |
| **Quality Metrics** | Answer depth and coverage | Option quality, clarity, and answerability |

---

## 📁 File Structure

benchmark/
├── .env # Environment file with API keys
├── MCQ/
│ ├── build_passage_score_mapping_per_file.py # Prepare data by grouping passages by score
│ ├── test_set_generator.py # Generate MCQ datasets under Base, Golden, and Mix settings
│ ├── generator_benchmark.py # Benchmark external and remote models (e.g., GPT, Gemini)
│ ├── local_model_evaluator.py # Evaluate local LLMs (e.g., LLaMA, Qwen, Gemma, Mistral)
│ └── README.md # This documentation
└── OE/
└── ... # Open-ended system

---

## 🚀 Features

### 1. MCQ Generation
- **High-quality Question Construction**: Converts human or LLM-generated queries into structured multiple-choice questions.
- **Balanced Distractors**: Creates 3 semantically related but incorrect options for robust model testing.
- **Three Information Settings**:
  - **Base**: No external passage (knowledge-only)
  - **Golden**: Gold-standard passage (score=3)
  - **Mix**: 30% low-quality passages to simulate noisy retrieval

### 2. Dataset Construction
- Automatically merges test queries and qrels.
- Groups evidence passages by score (0–3) for controlled retrieval quality.
- Generates standardized test sets with consistent structure.

### 3. Model Evaluation
- Supports both **cloud** (OpenAI, Gemini) and **local** (LLaMA, Qwen, Mistral, Gemma, etc.) evaluation.
- Handles 3000+ concurrent API calls for efficient benchmarking.
- Computes per-model accuracy and explanation correctness.

### 4. Quality Assessment
Evaluates question quality using GPT-based rubric:
- **Clarity** – Well-formed and unambiguous?
- **Difficulty** – Suitable challenge level?
- **Relevance** – Consistent with passage?
- **Educational Value** – Reflects meaningful understanding?
- **Option Quality** – Plausible distractors?
- **Cognitive Level** – Tests reasoning or fact recall?
- **Answerability** – Solvable using given passage?

---

## 🛠️ Setup

### 1. Environment
Create a `.env` file in the `benchmark/` directory:

```bash
cd benchmark/
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env

2. Dependencies

Install Python dependencies:
pip install openai python-dotenv tqdm torch transformers sentence-transformers scikit-learn nltk

📖 Usage
1. Generate MCQ Datasets
from test_set_generator import TestSetGenerator

generator = TestSetGenerator(
    data_prepare_dir="/home/shared/RAG_DATA/benchmark/data_prepare",
    output_dir="/home/shared/RAG_DATA/benchmark/generated_test_sets"
)
generator.initialize()
test_set = generator.generate_test_set(num_cases=20)

2. Run Cloud Model Benchmark
python generator_benchmark.py
This script runs all test sets (base, golden, mix) using models like GPT-4o or Gemini.

3. Run Local Model Evaluation
python local_model_evaluator.py --model qwen-3-8b --test_set /path/to/test_set_golden_simple.json
Supported models:

LLaMA 3 (8B)
Qwen 2.5 (3B/7B)
Mistral 3 (7B/8B)
Gemma 2 (2B/7B)
Phi (2/4)
DeepSeek V3 (7B)
TinyLLaMA (1.1B)

📊 Output Format
Example MCQ Item
{
  "original_query": "What is the main cause of urban flooding?",
  "passage": "Urban flooding is mainly caused by poor drainage and excessive rainfall.",
  "setting": "golden",
  "metadata": {
    "general_type": "qa",
    "quality_score": 3.0,
    "difficulty_level": "medium"
  },
  "multiple_choice": {
    "gpt40": {
      "content": {
        "question": "What factor contributes most to urban flooding?",
        "options": ["A. Deforestation", "B. Poor drainage", "C. Earthquakes", "D. Industrial emissions"],
        "correct_option": "B",
        "reason": "Urban flooding results from inadequate drainage systems."
      }
    }
  }
}

Quality Assessment Output
{
  "clarity": 5,
  "difficulty": 4,
  "relevance": 5,
  "educational_value": 4,
  "option_quality": 5,
  "cognitive_level": 4,
  "answerability": 5,
  "final_score": 4.6,
  "is_valid": true
}

🧩 Quality Evaluation Thresholds

A question is valid if:

Clarity ≥ 4

Relevance ≥ 4

Option Quality ≥ 4

Answerability ≥ 4

Final Score ≥ 4.0
🎯 Use Cases

RAG Benchmarking – Evaluate retrieval and reasoning under controlled settings.

Model Comparison – Compare accuracy across GPT, Gemini, and open models.

Educational QA Systems – Create realistic multi-option questions.

Dataset Construction – Produce reproducible, high-quality benchmark data.

🔧 Configuration
Parameter	Description	Default
batch_size	Number of questions saved per checkpoint	50
concurrency	Concurrent API calls	300 (OpenAI) / 500 (local)
temperature	Sampling temperature for generation	1e-5
top_k	Passages retrieved for Golden evaluation	5
🚨 Important Notes

Costs: Each batch may use several hundred thousand tokens for generation.

Reproducibility: All random seeds and API settings fixed for consistency.

Passage Replacement: Mix setting automatically replaces 30% of gold passages with low-score ones.

Checkpointing: Intermediate progress auto-saved every 50 items.

🤝 Comparison with OE System
Scenario	Recommended System
Factual reasoning, distractor quality	MCQ System
Deep reasoning, narrative completeness	OE System
Retrieval robustness benchmarking	Both (Base/Mix/Golden)
📞 Support

For debugging or extension:

Check the console logs for failed API calls.

Validate output structure using JSONLint.

Verify all .env API keys are set properly.

For model evaluation, ensure GPU memory ≥ 16 GB.
