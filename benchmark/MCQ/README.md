# Multiple-Choice (MCQ) Question Generation System

This directory contains the **Multiple-Choice Question (MCQ)** Generation System, designed to create, evaluate, and benchmark large-scale multiple-choice datasets for **RAG (Retrieval-Augmented Generation)** evaluation and model comparison.

---

## 🔄 Differences from OE System

| Aspect               | OE System                              | MCQ System                                           |
| -------------------- | -------------------------------------- | ---------------------------------------------------- |
| **Question Type**    | Open-ended questions with long answers | Multiple-choice questions (4 options)                |
| **Answer Format**    | 3–5 paragraph detailed answer          | Single correct letter (A/B/C/D)                      |
| **Evaluation Focus** | Comprehensiveness and reasoning        | Option quality, correctness, distractor plausibility |
| **Output Structure** | `open_ended.gpt40.content`             | `multiple_choice.gpt40.content`                      |
| **Quality Metrics**  | Answer depth and coverage              | Option quality, clarity, and answerability           |

---

## 📁 File Structure

```
benchmark/
├── .env                            # Environment file with API keys
├── MCQ/
│   ├── build_passage_score_mapping_per_file.py  # Group passages by score
│   ├── test_set_generator.py       # Generate MCQ datasets (Base, Golden, Mix)
│   ├── generator_benchmark.py      # Benchmark cloud models (GPT, Gemini)
│   ├── local_model_evaluator.py    # Evaluate local models (LLaMA, Qwen, etc.)
│   └── README.md                   # This documentation
└── OE/
    └── ...                         # Open-ended system
```

---

## 🛠️ Setup

### 1. Environment

```bash
cd benchmark/
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
```

### 2. Dependencies

```bash
pip install openai python-dotenv tqdm torch transformers sentence-transformers scikit-learn nltk
```

---

## 📖 Usage

### 1. Generate MCQ Datasets

```python
from test_set_generator import TestSetGenerator

generator = TestSetGenerator(
    data_prepare_dir="/home/shared/RAG_DATA/benchmark/data_prepare",
    output_dir="/home/shared/RAG_DATA/benchmark/generated_test_sets"
)
generator.initialize()
test_set = generator.generate_test_set(num_cases=20)
```

### 2. Run Cloud Model Benchmark

```bash
python generator_benchmark.py
```

This runs all test sets (`base`, `golden`, `mix`) using models such as GPT-4o or Gemini.

### 3. Run Local Model Evaluation

```bash
python local_model_evaluator.py --model qwen-3-8b --test_set /path/to/test_set_golden_simple.json
```

**Supported Models:**

* LLaMA 3 (8B)
* Qwen 2.5 (3B / 7B)
* Mistral 3 (7B / 8B)
* Gemma 2 (2B / 7B)
* Phi (2 / 4)
* DeepSeek V3 (7B)
* TinyLLaMA (1.1B)

---

## 📊 Output Format

### Example MCQ Item

```json
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
```

### Quality Assessment Output

```json
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
```

---

## 🧩 Quality Evaluation Thresholds

A question is valid if:

* Clarity ≥ 4
* Relevance ≥ 4
* Option Quality ≥ 4
* Answerability ≥ 4
* Final Score ≥ 4.0

---

## 🎯 Use Cases

* **RAG Benchmarking** – Evaluate retrieval and reasoning under controlled settings.
* **Model Comparison** – Compare accuracy across GPT, Gemini, and open models.
* **Educational QA Systems** – Create realistic multi-option questions.
* **Dataset Construction** – Produce reproducible, high-quality benchmark data.

---

## 🔧 Configuration

| Parameter     | Description                              | Default                    |
| ------------- | ---------------------------------------- | -------------------------- |
| `batch_size`  | Number of questions saved per checkpoint | 50                         |
| `concurrency` | Concurrent API calls                     | 300 (OpenAI) / 500 (local) |
| `temperature` | Sampling temperature for generation      | 1e-5                       |
| `top_k`       | Passages retrieved for Golden evaluation | 5                          |

---

## 🚨 Important Notes

1. **Costs:** Each batch may use several hundred thousand tokens for generation.
2. **Reproducibility:** All random seeds and API settings fixed for consistency.
3. **Passage Replacement:** Mix setting automatically replaces 30% of gold passages with low-score ones.
4. **Checkpointing:** Intermediate progress auto-saved every 50 items.

---

## 🤝 Comparison with OE System

| Scenario                               | Recommended System             |
| -------------------------------------- | ------------------------------ |
| Factual reasoning, distractor quality  | **MCQ System**                 |
| Deep reasoning, narrative completeness | **OE System**                  |
| Retrieval robustness benchmarking      | **Both (Base / Mix / Golden)** |

---

## 📞 Support

For debugging or extension:

1. Check console logs for API or parsing errors.
2. Validate output structure using JSONLint.
3. Verify `.env` API keys are properly configured.
4. For local evaluation, ensure GPU memory ≥ 16 GB.
