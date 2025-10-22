import json
import os
import argparse
from typing import List, Dict, Any
import openai
from tqdm import tqdm
# import anthropic
# import google.generativeai as genai
import requests
import time
import re
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import random
import asyncio
import aiohttp

class GeneratorBenchmark:
    # JSON Schema for structured output
    JSON_SCHEMA = {
        "type": "OBJECT",
        "properties": {
            "choice": {"type": "STRING", "enum": ["A","B","C","D"]},
            "reason": {"type": "STRING"}
        },
        "required": ["choice"]
    }

    PROMPT_TEMPLATE = """Question: {q}
Options:
{opts}
{passage_block}

Answer only the letter: A, B, C, or D""".strip()
    
    def __init__(self, test_set_path: str, model_name: str = None):
        """
        Initialize the generator benchmark
        
        Args:
            test_set_path: Path to the test set (base/golden/mix)
            model_name: Name of the model (for golden lookup optimization)
        """
        # Load .env file from DATA directory
        env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
        load_dotenv(dotenv_path=env_path)
        
        self.test_set = self._load_test_set(test_set_path)
        self.setting = os.path.basename(test_set_path).split('_')[0]  # base/golden/mix
        self.model_name = model_name
        
        # 如果是 mix 设置，加载 golden 结果作为查找表
        self.golden_lookup = {}
        if self.setting == "mix" and model_name:
            self._load_golden_lookup()
        
        # API keys
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        # self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        

        
        # Thread safety
        self.print_lock = Lock()
        self.results_lock = Lock()
        
        # Set random seed for reproducibility
        # random.seed(42)
        

    
    async def __aenter__(self):
        """Initialize async context"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up async context"""
        pass
    
    def _load_test_set(self, test_set_path: str) -> List[Dict]:
        """Load the test set"""
        with open(test_set_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_golden_lookup(self):
        """Load golden results as lookup table for mix optimization"""
        try:
            # 构建 golden 结果文件路径
            base_dir = "/home/shared/RAG_DATA"
            golden_path = f"{base_dir}/DATA/local_MCQ/{self.model_name}/golden_test.json"
            
            if os.path.exists(golden_path):
                print(f"Loading golden results from: {golden_path}")
                with open(golden_path, 'r', encoding='utf-8') as f:
                    golden_data = json.load(f)
                
                # 构建查找表
                for item in golden_data:
                    # 使用 question + passage 作为 key
                    question = item['question']
                    passage = item.get('passage', '')  # 从原始数据中获取 passage
                    key = f"{question}_{passage}"
                    self.golden_lookup[key] = {
                        'model_answer': item['model_answer'],
                        'model_explanation': item.get('model_explanation', ''),
                        'is_correct': item['is_correct'],
                        'correct_answer': item['correct_answer'],
                    }
                
                print(f"Loaded {len(self.golden_lookup)} golden results for lookup")
            else:
                print(f"Warning: Golden results not found at {golden_path}")
                print("Will process mix without optimization")
                
        except Exception as e:
            print(f"Error loading golden lookup: {e}")
            print("Will process mix without optimization")
    
    async def generate_answer(self, query: str, context: List[str], model: str, options: List[str]) -> str:
        """Generate answer for MCQ using structured JSON output"""
        # Build options string - remove original prefixes and re-number sequentially  
        clean_options = [opt.split('. ', 1)[1] if '. ' in opt else opt for opt in options]
        options_str = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(clean_options)])
        
        # Build passage block
        passage_block = ""
        if context:
            passage_block = "Passage:\n" + "\n".join(context)
        
        # Use the structured prompt template
        prompt = self.PROMPT_TEMPLATE.format(q=query, opts=options_str, passage_block=passage_block)
        
        # Call the appropriate model
        return await self._call_model(model, prompt)
    
    async def _call_model(self, model_name: str, prompt: str) -> str:
        """Call the appropriate model based on model name.
        
        Args:
            model_name: Name of the model to use
            prompt: The prompt to send to the model
            
        Returns:
            str: Model's response
        """
        if model_name in ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"]:  # Add gpt-4o to supported models
            return await self._call_openai(model_name, prompt)
        elif model_name in ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"]:  # Add Gemini models
            return await self._call_google(model_name, prompt)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    async def _call_openai(self, model: str, prompt: str) -> str:
        """Call OpenAI API asynchronously."""
        client = openai.AsyncOpenAI(api_key=self.openai_api_key)
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers multiple choice questions based on the given context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,  # Very low temperature for deterministic answers
                max_tokens=1000   # Add max_tokens parameter
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling OpenAI API for model {model}: {str(e)}")
            raise
    
    # def _call_anthropic(self, prompt: str) -> str:
    #     client = anthropic.Anthropic(api_key=self.anthropic_api_key)
    #     response = client.messages.create(
    #         model="claude-2",
    #         messages=[{"role": "user", "content": prompt}]
    #     )
    #     return response.content[0].text
    
    async def _call_google(self, model_name: str, prompt: str) -> str:
        """Call Google Gemini API using standard REST API."""
        max_retries = 5
        base = 1.0  # Start with 1 second base delay
        
        for attempt in range(max_retries):
            try:
                # 使用标准Generative Language API
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
                headers = {"Content-Type": "application/json"}
                params = {"key": self.google_api_key}
                
                data = {
                    "systemInstruction": {
                        "parts": [{"text": "You are a helpful assistant that answers multiple choice questions based on the given context."}]
                    },
                    "contents": [{
                        "role": "user",
                        "parts": [{"text": prompt}]
                    }],
                    "generationConfig": {
                        "temperature": 0.0,
                        "maxOutputTokens": 1000,  # 给Gemini 2.5 Pro的思考+输出足够空间，但不保存详细响应
                    },
                    "safetySettings": [
                        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"}
                    ]
                }
                
                # 检查API Key
                if not self.google_api_key:
                    raise Exception("Google API Key not found in environment variables")
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=data, headers=headers, params=params) as response:
                        if response.status == 200:
                            result = await response.json()
                            # Handle different response formats
                            if "candidates" in result and result["candidates"]:
                                candidate = result["candidates"][0]
                                finish_reason = candidate.get("finishReason", "")
                                
                                # Handle MAX_TOKENS case - return empty response
                                if finish_reason == "MAX_TOKENS":
                                    return "Unable to complete response due to token limit"
                                
                                if "content" in candidate:
                                    content = candidate["content"]
                                    if "parts" in content and content["parts"]:
                                        # 正常情况：有parts数组
                                        return content["parts"][0]["text"]
                                    elif "text" in content:
                                        # 备选：直接text字段
                                        return content["text"]
                                    elif content.get("role") == "model" and not content.get("parts"):
                                        # Gemini 2.5 Pro的特殊情况：content只有role，没有实际内容
                                        # 这通常意味着模型没有生成任何文本，可能是prompt问题
                                        return "Unable to generate response - empty model output"
                                elif "text" in candidate:
                                    return candidate["text"]
                                
                            # 如果没有返回，抛异常由重试兜底
                            raise Exception(f"Unexpected response format: {result}")
                        elif response.status in (429, 500, 502, 503, 504):
                            # 显式抛出可重试错误
                            text = await response.text()
                            raise RuntimeError(f"RETRYABLE {response.status}: {text}")
                        else:
                            # 不可重试
                            text = await response.text()
                            raise Exception(f"API call failed {response.status}: {text}")
                
            except Exception as e:
                msg = str(e).lower()
                retryable = isinstance(e, RuntimeError) or "quota" in msg or "rate" in msg or "limit" in msg
                if retryable and attempt < max_retries - 1:
                    # 对于429错误，使用更长的等待时间
                    if "429" in str(e) or "quota" in str(e).lower():
                        wait = 60 + random.uniform(0, 30)  # 等待60-90秒
                    else:
                        wait = base * (2 ** attempt) + random.uniform(0, 0.5)  # 其他错误正常退避
                    print(f"[Retry {attempt+1}/{max_retries}] {model_name}: {e} -> sleep {wait:.2f}s")
                    await asyncio.sleep(wait)
                    continue
                print(f"Error calling Google Gemini API for model {model_name}: {e}")
                raise
    

    
    def run_benchmark(self, model: str) -> Dict:
        """Run benchmark for a specific model"""
        results = {
            "predictions": [],
            "references": [],
            "correct": 0,
            "total": 0
        }
        
        for case in tqdm(self.test_set, desc=f"Running {model} on {self.setting}"):
            # Get question and options
            question = case["multiple_choice"]["gpt40"]["content"]["question"]
            options = case["multiple_choice"]["gpt40"]["content"]["options"]
            correct_answer = case["multiple_choice"]["gpt40"]["content"]["correct_option"]
            
            # Get context based on setting
            if self.setting == "base":
                context = []
            else:
                context = [case["passage"]] if "passage" in case else []
            
            # Generate answer
            answer = self.generate_answer(question, context, model, options)
            
            # Store results
            results["predictions"].append(answer)
            results["references"].append(correct_answer)
            results["total"] += 1
            
            # Check if correct
            if answer.strip().upper() == correct_answer.strip().upper():
                results["correct"] += 1
        
        # Calculate accuracy
        results["accuracy"] = results["correct"] / results["total"]
        
        return results

#     def _generate_prompt(self, item: Dict[str, Any]) -> str:
#         """Generate prompt for the model based on test set item."""
#         # Get question and options
#         if "multiple_choice" in item and "gpt40" in item["multiple_choice"]:
#             content = item["multiple_choice"]["gpt40"]["content"]
#             question = content["question"]
#             options = content["options"]
#         else:
#             raise ValueError("Invalid test set format: missing multiple_choice or gpt40 content")

#         # Get retrieved documents if available
#         retrieved_docs = item.get("retrieved_documents", [])
        
#         # Build prompt
#         prompt = f"""Please answer the following multiple choice question. Choose the most appropriate answer from the given options.

# Question: {question}

# Options:
# """
#         # Add options
#         for i, opt in enumerate(options):
#             prompt += f"{chr(65 + i)}. {opt.split('. ')[1]}\n"

#         if retrieved_docs:
#             prompt += "\nRelevant information:\n"
#             for i, doc in enumerate(retrieved_docs, 1):
#                 prompt += f"{i}. {doc}\n"
        
#         prompt += "\nPlease provide your answer in the following format:\nAnswer: [A/B/C/D]\nExplanation: [Your reasoning]"
        
#         return prompt

    def _parse_model_response(self, response: str) -> Dict[str, str]:
        """Parse model's response to extract answer and explanation."""
        # Try to find answer and explanation in the response
        answer = None
        explanation = None
        
        # Look for "Answer: X" pattern
        answer_match = re.search(r"Answer:\s*([A-D])", response, re.IGNORECASE)
        if answer_match:
            answer = answer_match.group(1)
        
        # Look for "Explanation:" or "Reasoning:" followed by text
        explanation_match = re.search(r"(?:Explanation|Reasoning):\s*(.*?)(?:\n|$)", response, re.DOTALL | re.IGNORECASE)
        if explanation_match:
            explanation = explanation_match.group(1).strip()
        
        return {
            "answer": answer,
            "explanation": explanation
        }

    async def process_single_item(self, item: Dict[str, Any], model_name: str, item_index: int, total: int) -> Dict[str, Any]:
        """Process a single test item."""
        try:
            # Get question and options
            question = item["multiple_choice"]["gpt40"]["content"]["question"]
            options = item["multiple_choice"]["gpt40"]["content"]["options"]
            correct_answer = item["multiple_choice"]["gpt40"]["content"]["correct_option"]
            
            # Get context based on setting
            if self.setting == "base":
                context = []
                passage = ""
            else:
                context = [item["passage"]] if "passage" in item else []
                passage = item.get("passage", "")
                if not context:
                    return None  # Skip items without passage for mix/golden settings
            
            # 检查是否可以从 golden 查找表中获取结果
            lookup_key = f"{question}_{passage}"
            if self.setting == "mix" and lookup_key in self.golden_lookup:
                # 直接使用 golden 结果
                golden_result = self.golden_lookup[lookup_key]
                result = {
                    "question": question,
                    "options": options,
                    "correct_answer": correct_answer,
                    "model_answer": golden_result['model_answer'],
                    "is_correct": golden_result['is_correct']
                }
                # 确保passage字段正确传递到结果中
                if passage:
                    result['passage'] = passage
                print(f"✅ Found in golden lookup: {question[:50]}...")
                return result
            
            # Generate answer
            raw_answer = await self.generate_answer(question, context, model_name, options)
            
            # Parse the answer and explanation
            answer, explanation = self._parse_response(raw_answer)
            if not answer:
                # If parsing fails, take the first A/B/C/D found in the response
                for char in raw_answer:
                    if char.upper() in ['A', 'B', 'C', 'D']:
                        answer = char.upper()
                        break
                if not answer:  # If still no answer found, randomly choose one
                    answer = random.choice(['A', 'B', 'C', 'D'])
            
            # Create result (removed model_explanation to save tokens)
            result = {
                "question": question,
                "options": options,
                "correct_answer": correct_answer,
                "model_answer": answer,
                "is_correct": answer == correct_answer
            }
            
            # 确保passage字段正确传递到结果中
            if passage:
                result['passage'] = passage
            
            # 添加original_query字段以便分类分析
            if 'original_query' in item:
                result['original_query'] = item['original_query']
            
            return result
            
        except Exception as e:
            print(f"[ERROR] item {item_index}/{total}: {e}")
            if 'raw_answer' in locals():
                print(f"Raw answer: {raw_answer}")
            else:
                print("Raw answer: N/A")
            return None

    def _parse_response(self, response: str) -> tuple[str, str]:
        """Parse the model's response to extract answer letter only.
        
        Args:
            response: The model's full response text
            
        Returns:
            tuple: (answer_letter, response)
        """
        resp = response.strip()
        
        # 1) 找第一个独立的 A/B/C/D（最常见情况）
        m = re.search(r'\b([A-D])\b', resp, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper(), resp
        
        # 2) 如果没找到，返回空答案
        return "", resp

    async def process_test_set(self, test_set: List[Dict[str, Any]], model_name: str, num_tests: int = None, concurrency: int = 500) -> List[Dict[str, Any]]:
        """Process test set with specified model using asyncio.
        
        Args:
            test_set: List of test cases
            model_name: Name of the model to use
            num_tests: Number of test cases to process. If None, process all.
            concurrency: Maximum number of concurrent requests
        """
        results = []
        total = len(test_set)
        
        # If num_tests is specified, randomly select that many cases
        if num_tests is not None:
            test_set = random.sample(test_set, min(num_tests, total))
            total = len(test_set)
            print(f"\nRandomly selected {total} items for testing...")
        else:
            print(f"\nProcessing all {total} items...")
        
        # Create semaphore to limit concurrency
        sem = asyncio.Semaphore(concurrency)
        
        async def process_with_semaphore(item, index):
            async with sem:
                return await self.process_single_item(item, model_name, index, total)
        
        # Create tasks
        tasks = [
            process_with_semaphore(item, i+1)
            for i, item in enumerate(test_set)
        ]
        
        # Process all tasks
        for result in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Processing {self.setting} test set"):
            result = await result
            if result is not None:
                results.append(result)
        
        # Calculate statistics
        total_answers = len(results)
        if total_answers == 0:
            print(f"\nNo valid results for {self.setting} setting")
            return results
            
        
        print(f"\nConfidence Statistics for {self.setting}:")
        print(f"Total processed: {total_answers}/{total} ({total_answers/total*100:.1f}%)")
        
        return results

def load_test_set(file_path: str) -> List[Dict[str, Any]]:
    """Load test set from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_results(results: List[Dict[str, Any]], output_path: str):
    """Save results to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def print_test_set_statistics(test_sets: List[str]):
    """Print statistics for all test sets."""
    print("\nTest set statistics:")
    for test_set in test_sets:
        with open(test_set, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Get all questions
            questions = [item["multiple_choice"]["gpt40"]["content"]["question"] for item in data]
            # Count unique questions
            unique_questions = set(questions)
            print(f"\n{os.path.basename(test_set)}:")
            print(f"Total items: {len(data)}")
            print(f"Unique questions: {len(unique_questions)}")
            # Print some example questions
            print("Example questions:")
            for q in list(unique_questions)[:3]:
                print(f"- {q}")

def main():
    # Specify the model to use
    # Options: "gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o", "gemini-2.5-pro", "gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"
    model_name = "gemini-1.5-pro"  # Change this to the model you want to use
    
    # Set number of test cases to process (None for all)
    num_tests = None  # Process ALL test cases (2000 per setting)
    
    # Set concurrency limit - optimal concurrency for Gemini 1.5 Pro
    # Based on testing, 500 concurrent seems optimal
    concurrency = 500 if model_name.startswith("gemini") else 3000  # Optimal 500 concurrent
    
    # Set paths
    test_sets = [
        "/home/shared/RAG_DATA/DATA/final_mcq/base_2000.json",
        "/home/shared/RAG_DATA/DATA/final_mcq/golden_2000.json",  # 如果有
        "/home/shared/RAG_DATA/DATA/final_mcq/mix_2000.json"
    ]
    
    # Create output directory
    output_dir = f"/home/shared/RAG_DATA/DATA/local_MCQ/{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Print statistics for all test sets
    print_test_set_statistics(test_sets)
    
    print(f"\n{'='*50}")
    print(f"Starting benchmark with {num_tests if num_tests else 'all'} test cases")
    print(f"{'='*50}\n")
    
    # Run benchmarks
    async def run_benchmarks():
        for test_set in test_sets:
            setting = os.path.basename(test_set).split('_')[0]  # base/golden/mix
            output_path = os.path.join(output_dir, f"{setting}_test.json")
            
            # Check if output file already exists
            if os.path.exists(output_path):
                print(f"\nOutput file for {setting} setting already exists at: {output_path}")
                print("Skipping this setting...")
                continue
            
            print(f"\nTesting {model_name} on {setting} setting...")
            
            async with GeneratorBenchmark(test_set, model_name) as benchmark:
                results = await benchmark.process_test_set(benchmark.test_set, model_name, num_tests, concurrency)
                
                # Save results for this setting
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                
                print(f"\nResults for {setting} saved to: {output_path}")
    
    # Run the async main function
    asyncio.run(run_benchmarks())

if __name__ == "__main__":
    main()