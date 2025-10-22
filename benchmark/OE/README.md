# Open-Ended (OE) Question Generation System

This directory contains the Open-Ended Question Generation System, which is specifically designed to generate comprehensive open-ended questions and detailed answers for RAG (Retrieval-Augmented Generation) evaluation.

## 🔄 Differences from MCQ System

| Aspect | MCQ System | OE System |
|--------|------------|-----------|
| **Question Type** | Multiple choice with 4 options | Open-ended questions requiring detailed answers |
| **Answer Format** | Single letter (A/B/C/D) | Comprehensive paragraphs (3-5 paragraphs) |
| **Evaluation Focus** | Option quality, distractors | Answer comprehensiveness, depth |
| **Output Structure** | `multiple_choice.gpt40.content` | `open_ended.gpt40.content` |
| **Quality Metrics** | Option quality, correct/incorrect | Answer quality, completeness |

## 📁 File Structure

```
benchmark/
├── .env                    # API keys (create this file)
├── OE/
│   ├── generate_oe_set.py      # Main OE generation system
│   ├── test_oe_generation.py   # Test script for validation
│   └── README.md               # This documentation
└── MCQ/
    └── ...                     # MCQ system files
```

## 🚀 Features

### 1. Open-Ended Question Generation
- **Enhanced Questions**: Refines user queries into clear, comprehensive questions
- **Detailed Answers**: Generates 3-5 paragraph answers with evidence from reference documents
- **Key Points Extraction**: Identifies main points covered in the answer
- **Reasoning**: Provides explanation of how the answer addresses the question

### 2. Quality Assessment
Evaluates questions and answers on 7 dimensions:
- **Clarity**: How clear and well-formulated is the question?
- **Difficulty**: How challenging is the question?
- **Relevance**: How relevant to the reference document?
- **Educational Value**: How well does it test understanding?
- **Answer Quality**: How comprehensive and well-structured is the answer?
- **Cognitive Level**: Does it test higher-order thinking?
- **Answerability**: Can it be answered using the reference document?

### 3. Multi-Setting Generation
- **Base**: Questions without reference passages
- **Golden**: Questions with high-quality (score=3) passages
- **Mix**: Questions with mixed-quality passages (30% replaced with lower-quality)

## 🛠️ Setup

### 1. Environment Setup
Create a `.env` file in the `benchmark/` directory:
```bash
cd benchmark/
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
```

### 2. Dependencies
Make sure you have the required Python packages:
```bash
pip install openai python-dotenv tqdm sentence-transformers scikit-learn nltk
```

## 📖 Usage

### Basic Usage
```python
from generate_oe_set import TestSetGenerator

# Initialize the generator
generator = TestSetGenerator(
    data_prepare_dir="path/to/your/data",
    output_dir="path/to/output"
)

# Initialize and load data
generator.initialize()

# Generate test cases
test_set = generator.generate_test_set(num_cases=10)
```

### Test the System
```bash
cd benchmark/OE
python test_oe_generation.py
```

## 📊 Output Format

### Question-Answer Structure
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
        "question": "Enhanced version of the original question...",
        "answer": "Comprehensive 3-5 paragraph answer...",
        "key_points": ["Point 1", "Point 2", "Point 3"],
        "reasoning": "Explanation of how the answer addresses the question"
      }
    }
  }
}
```

### Quality Assessment
```json
{
  "clarity": 5,
  "difficulty": 4,
  "relevance": 5,
  "educational_value": 4,
  "answer_quality": 5,
  "cognitive_level": 4,
  "answerability": 5,
  "explanation": "Detailed quality assessment...",
  "final_score": 4.6,
  "is_valid": true
}
```

## 🎯 Use Cases

1. **RAG System Evaluation**: Generate comprehensive test sets for evaluating retrieval-augmented generation systems
2. **Educational Assessment**: Create open-ended questions for testing deep understanding
3. **Content Quality Evaluation**: Assess the quality of generated answers against reference documents
4. **Benchmark Creation**: Build standardized datasets for comparing different AI systems

## 🔧 Configuration

### Quality Thresholds
Questions are considered valid if they meet these minimum thresholds:
- Clarity ≥ 4
- Relevance ≥ 4  
- Answer Quality ≥ 4
- Answerability ≥ 4
- Final Score ≥ 4.0

### Batch Processing
- Batch size: 50 test cases per save operation
- Thread pool: Up to 300 concurrent workers
- API rate limiting: 500 concurrent requests

## 🚨 Important Notes

1. **API Costs**: Open-ended generation typically requires more tokens than MCQ generation
2. **Processing Time**: Answer generation takes longer than multiple choice options
3. **Quality Focus**: Emphasis on answer comprehensiveness and depth rather than correctness of options
4. **Manual Review**: Consider manual review of generated answers for critical applications

## 🤝 Comparison with MCQ

Choose **OE System** when you need:
- Detailed, comprehensive answers
- Assessment of reasoning and explanation abilities
- Evaluation of synthesis and analysis skills
- Open-ended response generation testing

Choose **MCQ System** when you need:
- Quick, objective assessment
- Large-scale automated evaluation
- Testing of specific facts or concepts
- Reduced evaluation complexity

## 📞 Support

For issues or questions about the OE generation system, please check:
1. Console output for detailed error messages
2. Quality assessment scores for validation
3. Generated test cases for format verification 