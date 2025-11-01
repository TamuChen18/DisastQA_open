# 🧠 DisastQA: Disaster-Domain Question Answering Benchmark

DisastQA is the **first large-scale, retrieval-aware benchmark** for factual and reliability-oriented QA in disaster response.  
It evaluates LLMs under **Base, Mix, and Golden** evidence settings across **8 disaster types**,  
with **MCQ (2,000)** and **OE (1,000)** human-validated test samples.

---

## 🚀 Key Features
- 🧩 **Dual-format Evaluation**: Multiple-Choice (MCQ) + Open-Ended (OE) QA  
- 📚 **Retrieval Control**: Base (no evidence) · Mix (partial) · Golden (fully grounded)  
- 👩‍💻 **Human–LLM Collaboration**: Expert-validated question rewriting and keypoint coverage  
- 🔍 **Multi-model Benchmarking**: Supports GPT, Gemini, LLaMA, Qwen, Mistral, Gemma, Phi, DeepSeek  
- 📊 **Multi-metric Evaluation**: Accuracy · Completeness · Keypoint Recall · Semantic Similarity  

---

## 🧱 Repository Structure
```
DisastQA/
├── data/              # Final datasets (small demo only)
├── code/              # Data generation, evaluation, and analysis scripts
├── results/           # Processed metrics and summary tables
├── model_results/     # Model-wise evaluation outputs
├── docs/              # Methodology, setup, and experiment notes
└── README.md          # Project overview (this file)
```

---

## ⚙️ Setup
```bash
pip install -r requirements.txt
echo "OPENAI_API_KEY=your_api_key" > .env
```

---

## 🧪 Quick Start
**Run local model benchmark**
```bash
python code/evaluation/mcq_evaluation/local_evaluation.py     --model qwen-3-8b     --test_set data/final_mcq/golden_2000.json
```

**Run closed model benchmark**
```bash
python code/evaluation/mcq_evaluation/evaluate_closemodel.py
```

---

## 📊 Output Example
```json
{
  "question": "What factor contributes most to urban flooding?",
  "options": ["A. Deforestation", "B. Poor drainage", "C. Earthquakes", "D. Industrial emissions"],
  "correct_answer": "B",
  "model_answer": "B",
  "is_correct": true,
  "explanation": "Poor drainage systems cause urban flooding."
}
```

---

## 📁 Documentation
- `docs/methodology.md`: Dataset & pipeline  
- `docs/experimental_setup.md`: Experiment design  
- `docs/results_interpretation.md`: Key findings  

---

## 📜 Citation
```
@dataset{chen2025disastqa,
  title = {DisastQA: Reliability-Oriented Disaster-Domain QA Benchmark},
  author = {Chen, Zhitong and Mostafavi, Ali and Caverlee, James},
  year = {2025},
  url = {https://github.com/zhitong-chen/DisastQA}
}
```
