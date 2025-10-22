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
    # load .env file
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    load_dotenv(dotenv_path=env_path)
    
    def __init__(self, test_set_path: str, model_name: str = None):
        """
        Initialize the generator benchmark
        
        Args:
            test_set_path: Path to the test set (base/golden/mix)
            model_name: Name of the model (for golden lookup optimization)
        """
        self.test_set = self._load_test_set(test_set_path)
        self.setting = os.path.basename(test_set_path).split('_')[2]  # base/golden/mix
        self.model_name = model_name
        
        # 如果是 mix 设置，加载 golden 结果作为查找表
        self.golden_lookup = {}
        if self.setting == "mix" and model_name:
            self._load_golden_lookup()
        
        # API keys
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        # self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        
        # open source models
        self.local_models = {
            "meta-llama/Meta-Llama-3-8B": "http://localhost:10000/v1",
            "mistral-3-7B": "http://localhost:10001/v1",
            "qwen-3-8B": "http://localhost:10002/v1",
            "deepseek-v3-7B": "http://localhost:10003/v1",
            "Phi-2": "http://localhost:10004/v1",
            "Gemma-7B": "http://localhost:10005/v1"
        }
        
        # Thread safety
        self.print_lock = Lock()
        self.results_lock = Lock()
        
        # Set random seed for reproducibility
        # random.seed(42)
        
        # Create aiohttp session for local models
        self.session = None
    
    async def __aenter__(self):
        """Initialize aiohttp session when entering async context"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close aiohttp session when exiting async context"""
        if self.session:
            await self.session.close()
    
    def _load_test_set(self, test_set_path: str) -> List[Dict]:
        """Load the test set"""
        with open(test_set_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_golden_lookup(self):
        """Load golden results as lookup table for mix optimization"""
        try:
            # 构建 golden 结果文件路径
            base_dir = "/home/shared/RAG_DATA"
            golden_path = f"{base_dir}/benchmark/testQA_set/{self.model_name}/golden_results.json"
            
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
        """Generate answer for MCQ"""
        # Build context string
        context_str = ""
        if context:  # For mix and golden settings
            context_str = "\n".join(context)
        
        # Build options string
        options_str = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])  # A, B, C, D
        
        # Build prompt based on setting
        if self.setting == "base":
            prompt = f"""
            Answer the following multiple choice question based on your knowledge.
            You MUST choose exactly one option from A, B, C, or D.

            Question: {query}

            Options:
            {options_str}

            Please provide your answer in the following format:
            Answer: [A/B/C/D]
            Explanation: [Your detailed reasoning for choosing this answer]

            Your explanation should:
            1. Explain why you chose this answer
            2. Explain why other options are incorrect
            3. Be based on your knowledge and understanding
            """
        else:  # mix and golden settings
            prompt = f"""
            You are given a reference passage and a multiple choice question.
            While you should primarily use the information from the passage to answer the question,
            you can also use your knowledge to help understand and interpret the passage.

            Reference Passage:
            {context_str}

            Question: {query}

            Options:
            {options_str}

            Please provide your answer in the following format:
            Answer: [A/B/C/D]
            Explanation: [Your detailed reasoning for choosing this answer]

            Your explanation should:
            1. First, quote and explain the relevant parts of the passage that support your answer
            2. Then, use your knowledge to help interpret and explain why this answer is correct
            3. Explain why other options are incorrect, using both passage information and your knowledge
            4. If the passage is unclear or ambiguous, use your knowledge to help resolve the ambiguity


            """
        
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
        elif model_name in ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"]:  # Add Gemini models
            return await self._call_google(model_name, prompt)
        elif model_name in self.local_models:
            return await self._call_local_model(model_name, prompt)
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
    
    # async def _call_google(self, model_name: str, prompt: str) -> str:
    #     """Call Google Gemini API asynchronously with retry logic."""
    #     max_retries = 3
    #     retry_delay = 1  # Start with 1 second delay
        
    #     for attempt in range(max_retries):
    #         try:
    #             genai.configure(api_key=self.google_api_key)
    #             model = genai.GenerativeModel(model_name)
                
    #             # Configure generation parameters for more deterministic responses
    #             generation_config = genai.types.GenerationConfig(
    #                 temperature=0.0,  # Very low temperature for deterministic answers
    #                 max_output_tokens=1000,  # Limit response length
    #             )
                
    #             # Use async generate_content
    #             response = await model.generate_content_async(
    #                 prompt,
    #                 generation_config=generation_config
    #             )
    #             return response.text
                
    #         except Exception as e:
    #             error_str = str(e).lower()
    #             # Check for rate limit or quota exceeded errors
    #             if ("quota" in error_str or "rate" in error_str or "limit" in error_str) and attempt < max_retries - 1:
    #                 wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
    #                 print(f"Rate limit hit for {model_name}, waiting {wait_time}s before retry...")
    #                 await asyncio.sleep(wait_time)
    #                 continue
    #             else:
    #                 print(f"Error calling Google Gemini API for model {model_name}: {str(e)}")
    #                 raise
    
    # async def _call_local_model(self, model: str, prompt: str) -> str:
    #     """Call local model API asynchronously."""
    #     if not self.session:
    #         raise RuntimeError("aiohttp session not initialized. Use async with context manager.")
            
    #     url = f"{self.local_models[model]}/chat/completions"
    #     try:
    #         async with self.session.post(url, json={
    #             "model": model,
    #             "messages": [
    #                 {"role": "system", "content": "You are a helpful assistant that answers multiple choice questions based on the given context."},
    #                 {"role": "user", "content": prompt}
    #             ]
    #         }) as response:
    #             if response.status != 200:
    #                 error_text = await response.text()
    #                 raise Exception(f"Local model API error: {error_text}")
    #             result = await response.json()
    #             return result["choices"][0]["message"]["content"]
    #     except Exception as e:
    #         print(f"Error calling local model {model}: {str(e)}")
    #         raise
    
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
                    "model_answer": golden_result['model_answer'],
                    "correct_answer": correct_answer,
                    "model_explanation": golden_result['model_explanation'],
                    "is_correct": golden_result['is_correct']
                }
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
            
            # Create result
            result = {
                "question": question,
                "options": options,
                "model_answer": answer,
                "correct_answer": correct_answer,
                "model_explanation": explanation or raw_answer,  # Use parsed explanation or full response
                "is_correct": answer == correct_answer
            }
            
            return result
            
        except Exception as e:
            return None

    def _parse_response(self, response: str) -> tuple[str, str, str]:
        """Parse the model's response to extract answer and explanation.
        
        Args:
            response: The model's full response text
            
        Returns:
            tuple: (answer_letter, explanation)
        """
        # Clean the response
        response = response.strip()
        
        # Try to find answer and explanation
        answer_match = re.search(r"Answer:\s*([A-D])", response, re.IGNORECASE)
        explanation_match = re.search(r"Explanation:\s*(.*?)(?:\n|$)", response, re.DOTALL | re.IGNORECASE)
        
        answer = answer_match.group(1).upper() if answer_match else ""
        explanation = explanation_match.group(1).strip() if explanation_match else ""
        
        # If no explanation found, try to get everything after the answer   
        if not explanation:
            explanation = response[answer_match.end():].strip()
        
        return answer, explanation

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
    # Options: "gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o", "gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"
    model_name = "gpt-3.5-turbo"  # Change this to the model you want to use
    
    # Set number of test cases to process (None for all)
    num_tests = None  # Process all test cases
    
    # Set concurrency limit (Gemini has rate limits, so use lower concurrency)
    concurrency = 100 if model_name.startswith("gemini") else 3000  # Lower for Gemini API
    
    # Set paths
    test_sets = [
        "/home/shared/RAG_DATA/benchmark/MCQ/generated_test_sets/test_set_base_simple.json",
        "/home/shared/RAG_DATA/benchmark/MCQ/generated_test_sets/test_set_golden_simple.json",
        "/home/shared/RAG_DATA/benchmark/MCQ/generated_test_sets/test_set_mix_simple.json"
    ]
    
    # Create output directory
    output_dir = f"/home/shared/RAG_DATA/benchmark/testMCQ_set/{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Print statistics for all test sets
    print_test_set_statistics(test_sets)
    
    print(f"\n{'='*50}")
    print(f"Starting benchmark with {num_tests if num_tests else 'all'} test cases")
    print(f"{'='*50}\n")
    
    # Run benchmarks
    async def run_benchmarks():
        for test_set in test_sets:
            setting = os.path.basename(test_set).split('_')[2]  # base/golden/mix
            print(f"\nTesting {model_name} on {setting} setting...")
            
            async with GeneratorBenchmark(test_set, model_name) as benchmark:
                results = await benchmark.process_test_set(benchmark.test_set, model_name, num_tests, concurrency)
                
                # Save results for this setting
                output_path = os.path.join(output_dir, f"{setting}_results.json")
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                
                print(f"\nResults for {setting} saved to: {output_path}")
    
    # Run the async main function
    asyncio.run(run_benchmarks())

if __name__ == "__main__":
    main()