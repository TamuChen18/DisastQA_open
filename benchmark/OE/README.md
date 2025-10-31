# Open-Ended (OE) Question Generation System

This module generates **open-ended questions** and **comprehensive answers** for evaluating Retrieval-Augmented Generation (RAG) systems.  
It complements the Multiple-Choice (MCQ) generation pipeline by focusing on **factual completeness**, **reasoning depth**, and **answer comprehensiveness**.

---

## Overview: MCQ vs. OE

| Aspect | MCQ System | OE System |
|--------|-------------|------------|
| **Question Type** | Multiple-choice (4 options) | Open-ended, paragraph-level responses |
| **Answer Format** | Single letter (A/B/C/D) | Multi-paragraph text (3–5 paragraphs) |
| **Evaluation Focus** | Option plausibility, distractors | Factual completeness, reasoning depth |
| **Output Structure** | `multiple_choice.gpt40.content` | `open_ended.gpt40.content` |
| **Quality Metrics** | Option quality, correctness | Clarity, relevance, answer quality, completeness |

---

## Directory Structure

```
benchmark/
├── .env                    # API keys (create this manually)
├── OE/
│   ├── generate_oe_set.py      # Core OE generation logic
│   ├── test_oe_generation.py   # Test script for validation
│   └── README.md               # This documentation
└── MCQ/
    └── ...                     # MCQ generation system
```

---

## Core Capabilities

### 1. Open-Ended QA Generation
- **Enhanced Question Refinement**: Converts raw user queries into clear, contextualized questions.  
- **Comprehensive Answer Synthesis**: Generates 3–5 paragraph answers grounded in evidence.  
- **Keypoint Extraction**: Identifies factual elements covered in the answer.  
- **Reasoning Explanation**: Provides rationale for how the answer addresses the question.

### 2. Quality Assessment Framework
Each QA pair is automatically evaluated across **seven dimensions**:

| Dimension | Description |
|------------|--------------|
| **Clarity** | Is the question well-phrased and unambiguous? |
| **Difficulty** | How conceptually challenging is it? |
| **Relevance** | Is it grounded in the given passage? |
| **Educational Value** | Does it test higher-order understanding? |
| **Answer Quality** | Is the answer comprehensive and coherent? |
| **Cognitive Level** | Does it require reasoning or synthesis? |
| **Answerability** | Can the passage support a correct answer? |

### 3. Multi-Setting Generation
| Setting | Description |
|----------|--------------|
| **Base** | Questions generated without any reference passages. |
| **Golden** | Questions generated from high-quality (score=3) passages. |
| **Mix** | Mixed set with ~30% of passages replaced by lower-quality ones. |

---

## Setup Instructions

### 1. Environment Configuration
Create a `.env` file in the benchmark directory:
```bash
cd benchmark/
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
```

### 2. Dependencies
```bash
pip install openai python-dotenv tqdm sentence-transformers scikit-learn nltk
```

---

## Example Usage

### Basic Generation
```python
from generate_oe_set import TestSetGenerator

generator = TestSetGenerator(
    data_prepare_dir="path/to/data",
    output_dir="path/to/output"
)

generator.initialize()
test_set = generator.generate_test_set(num_cases=10)
```

### Command-Line Test
```bash
cd benchmark/OE
python test_oe_generation.py
```

---

## Output Format

### QA Example
```json
{
  "original_query": "What are the main benefits of renewable energy?",
  "passage": "Reference document text...",
  "setting": "golden",
  "metadata": {
    "general_type": "qa",
    "quality_score": 3.0,
    "difficulty_level": "medium"
  },
  "open_ended": {
    "gpt40": {
      "content": {
        "question": "Enhanced question version...",
        "answer": "Comprehensive 3–5 paragraph answer...",
        "key_points": ["Point 1", "Point 2", "Point 3"],
        "reasoning": "Explanation of how the answer addresses the question"
      }
    }
  }
}
```

### Quality Assessment Example
```json
{
  "clarity": 5,
  "difficulty": 4,
  "relevance": 5,
  "educational_value": 4,
  "answer_quality": 5,
  "cognitive_level": 4,
  "answerability": 5,
  "final_score": 4.6,
  "is_valid": true
}
```

---

## Application Scenarios

- **RAG Evaluation** – Benchmark retrieval and generation under different context settings.  
- **Educational Assessment** – Design open-ended comprehension or reasoning tests.  
- **Answer Quality Auditing** – Assess factual completeness in model outputs.  
- **Benchmark Dataset Creation** – Build standardized OE evaluation datasets.

---

## Configuration Highlights

### Quality Thresholds
Questions are marked *valid* if:
- Clarity ≥ 4  
- Relevance ≥ 4  
- Answer Quality ≥ 4  
- Answerability ≥ 4  
- Final Score ≥ 4.0

### Batch Parameters
| Parameter | Value |
|------------|--------|
| **Batch Size** | 50 samples per save |
| **Thread Pool** | 300 concurrent workers |
| **API Rate Limit** | 500 concurrent requests |

---

## Notes

1. **API Costs** – OE generation consumes significantly more tokens than MCQ.  
2. **Longer Runtime** – Paragraph-level generation takes 2–3× longer per sample.  
3. **Manual Review Recommended** – For critical datasets, human curation improves reliability.  
4. **Focus on Depth** – Prioritizes reasoning and completeness, not short factual accuracy.

---

## Choosing Between MCQ and OE

| Use Case | Recommended System |
|-----------|--------------------|
| Objective factual checks | MCQ |
| Large-scale automatic grading | MCQ |
| Reasoning and synthesis testing | OE |
| Realistic RAG evaluation | OE |
| Human-level comprehension benchmarks | OE |

---

## Support
If you encounter issues:
1. Review console logs for error messages.  
2. Check the quality assessment scores.  
3. Inspect generated test cases for structural validity.
