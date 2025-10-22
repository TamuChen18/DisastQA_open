import json
import os
import argparse
from typing import List, Dict, Any
import openai
from tqdm import tqdm
import requests
import time
import re
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import random
import asyncio
import aiohttp

# 根据难度调整的token配置 - 基于key point涵盖率分析优化
DIFFICULTY_TOKEN_CONFIG = {
    "easy": {
        "max_new_tokens": 80,  # 简单问题，短答案 (100%涵盖率，无需调整)
        "word_limit": "10-100"
    },
    "medium": {
        "max_new_tokens": 180,  # 中等问题，中等长度答案 (100%涵盖率，无需调整)
        "word_limit": "20-200"
    },
    "hard": {
        "max_new_tokens": 300,  # 困难问题，增加token以提高涵盖率 (当前66.7%涵盖率)
        "word_limit": "60-350"  # 增加长度范围以覆盖更多key points
    },
    "extremely_complex": {
        "max_new_tokens": 400,  # 极复杂问题，大幅增加token (当前66.7%涵盖率)
        "word_limit": "120-500"  # 增加长度范围以覆盖更多key points
    }
}

# 闭源模型配置
CLOSED_MODEL_CONFIGS = {
    "gpt-4o": {
        "temperature": 0.7,
        "max_tokens": 500,
        "model_type": "openai"
    },
    "gpt-4o-mini": {
        "temperature": 0.7,
        "max_tokens": 500,
        "model_type": "openai"
    },
    "gpt-3.5-turbo": {
        "temperature": 0.7,
        "max_tokens": 500,
        "model_type": "openai"
    },
    "gemini-2.5-pro": {
        "temperature": 0.7,
        "max_tokens": 500,
        "model_type": "google"
    },
    "gemini-2.5-flash": {
        "temperature": 0.7,
        "max_tokens": 500,
        "model_type": "google"
    },
    "gemini-1.5-pro": {
        "temperature": 0.7,
        "max_tokens": 500,
        "model_type": "google"
    },
    "gemini-1.5-flash": {
        "temperature": 0.7,
        "max_tokens": 500,
        "model_type": "google"
    }
}

class ClosedModelOEEvaluator:
    """Evaluator for closed models (GPT, Gemini) on Open-Ended questions with difficulty-aware generation"""
    
    def __init__(self, test_set_path: str, model_name: str):
        """Initialize evaluator"""
        # Load .env file from DATA directory
        env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
        load_dotenv(dotenv_path=env_path)
        
        self.test_set_path = test_set_path
        self.model_name = model_name
        self.model_config = CLOSED_MODEL_CONFIGS[model_name]
        self.setting = os.path.basename(test_set_path).split('_')[0]  # base/golden/mix
        
        # 如果是 mix 设置，加载 golden 结果作为查找表
        self.golden_lookup = {}
        if self.setting == "mix":
            self._load_golden_lookup()
        
        # API keys
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        
        # Thread safety
        self.print_lock = Lock()
        self.results_lock = Lock()
        
        # Load test set
        with open(test_set_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # Convert data format for OE with difficulty
        self.test_set = []
        for item in raw_data:
            if 'open_ended' in item and 'gpt40' in item['open_ended']:
                content = item['open_ended']['gpt40']['content']
                passage = item.get('passage', '')  # 保存原始 passage
                difficulty = item.get('difficulty', 'medium')  # 获取difficulty，默认为medium
                processed_item = {
                    'question': content['question'],
                    'correct_answer': content['correct_answer'],  # OE的正确答案
                    'llm_answer': content.get('llm_answer', ''),  # OE的LLM答案
                    'context': [passage] if self.setting in ['golden', 'mix'] and passage else [],  # Use passage only in golden and mix settings
                    'passage': passage,  # 保存 passage 用于查找
                    'difficulty': difficulty  # 添加difficulty字段
                }
                self.test_set.append(processed_item)
        
        # 统计difficulty分布
        difficulty_counts = {}
        for item in self.test_set:
            difficulty = item['difficulty']
            difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
        
        print(f"\nDifficulty distribution in test set:")
        for difficulty, count in difficulty_counts.items():
            print(f"  {difficulty}: {count} questions")
        
        # Verify data format
        required_keys = ['question', 'correct_answer', 'difficulty']
        for i, item in enumerate(self.test_set):
            missing_keys = [key for key in required_keys if key not in item]
            if missing_keys:
                print(f"Warning: Item {i} is missing keys: {missing_keys}")

    def _load_golden_lookup(self):
        """Load golden results as lookup table for mix optimization"""
        try:
            # 构建 golden 结果文件路径
            base_dir = "/home/shared/RAG_DATA"
            golden_path = f"{base_dir}/DATA/local_OE/{self.model_name}/golden_oe_with_difficulty.json"
            
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
                        'correct_answer': item['correct_answer'],
                        'difficulty': item.get('difficulty', 'medium')
                    }
                
                print(f"Loaded {len(self.golden_lookup)} golden results for lookup")
            else:
                print(f"Warning: Golden results not found at {golden_path}")
                print("Will process mix without optimization")
                
        except Exception as e:
            print(f"Error loading golden lookup: {e}")
            print("Will process mix without optimization")

    def _get_difficulty_config(self, difficulty: str) -> Dict[str, Any]:
        """Get difficulty-specific configuration"""
        return DIFFICULTY_TOKEN_CONFIG.get(difficulty, DIFFICULTY_TOKEN_CONFIG["medium"])

    def _clean_response(self, response: str) -> str:
        """清理响应，移除格式说明和无关内容，并控制长度"""
        if not response:
            return response
        
        # 移除常见的格式说明
        format_patterns = [
            r'The answer should be in English[^.]*\.',
            r'Please do not use any markdown[^.]*\.',
            r'Please ensure that the answer is[^.]*\.',
            r'Please type the answer here[^.]*\.',
            r'Answer:\s*\([^)]*\)',  # 移除 "Answer: (Please type...)"
            r'Okay, so[^.]*\.',  # 移除 "Okay, so..."
            r'Let me[^.]*\.',  # 移除 "Let me..."
            r'I will[^.]*\.',  # 移除 "I will..."
            r'Based on[^.]*\.',  # 移除 "Based on..."
            r'\(Source:[^)]*\)',  # 移除 "(Source: ...)"
            r'\(Word Count:[^)]*\)',  # 移除 "(Word Count: ...)"
            r'\(Time[^)]*\)',  # 移除 "(Time...)"
            r'\[[0-9]+\]',  # 移除 "[1] [2] [3]" 等引用标记
        ]
        
        cleaned_response = response
        for pattern in format_patterns:
            cleaned_response = re.sub(pattern, '', cleaned_response, flags=re.IGNORECASE | re.DOTALL)
        
        # 移除多余的空白行和空格
        cleaned_response = re.sub(r'\n\s*\n', '\n', cleaned_response)
        cleaned_response = re.sub(r'^\s+|\s+$', '', cleaned_response, flags=re.MULTILINE)
        
        # 如果清理后为空，返回原始响应
        if not cleaned_response.strip():
            return response.strip()
        
        # 记录长度信息，但不截断
        words = cleaned_response.split()
        if len(words) > 500:  # 提高警告阈值，适应新的extremely_complex配置
            print(f"⚠️  Warning: Answer is quite long ({len(words)} words). Consider adjusting prompt or token limits.")
        
        return cleaned_response.strip()

    async def generate_answer(self, query: str, context: List[str], difficulty: str = "medium") -> str:
        """Generate answer for an open-ended question with difficulty-aware generation"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                context_str = "\n".join(context) if context else ""
                
                # 获取difficulty配置
                difficulty_config = self._get_difficulty_config(difficulty)
                word_limit = difficulty_config["word_limit"]
                
                # Build optimized prompt for OE questions with difficulty-aware length constraint
                if context_str:
                    prompt = f"""Passage: {context_str}

Question: {query}

Difficulty Level: {difficulty.upper()}

Instructions: Provide a comprehensive answer within {word_limit} words. Ensure you cover all important aspects and key points related to the question. Be thorough but concise.

Answer:"""
                else:
                    prompt = f"""Question: {query}

Difficulty Level: {difficulty.upper()}

Instructions: Provide a comprehensive answer within {word_limit} words. Ensure you cover all important aspects and key points related to the question. Be thorough but concise.

Answer:"""
                
                print(f"\nGenerating response for {difficulty} difficulty...")
                print(f"Prompt length: {len(prompt)}")
                
                # Call the appropriate model
                response = await self._call_model(prompt)
                
                # Clean up the response
                response = self._clean_response(response)
                
                print("Generated response:")
                print(response)
                
                if not response:
                    print("Warning: Empty response generated!")
                    if attempt < max_retries - 1:
                        print(f"Retrying... (attempt {attempt + 1}/{max_retries})")
                        continue
                    else:
                        raise ValueError("Failed to generate non-empty response after all retries")
                
                return response
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                print(f"\nAttempt {attempt + 1} failed: {str(e)}")
                await asyncio.sleep(1)  # Wait for 1 second before retrying

    async def _call_model(self, prompt: str) -> str:
        """Call the appropriate model based on model type."""
        if self.model_config["model_type"] == "openai":
            return await self._call_openai(prompt)
        elif self.model_config["model_type"] == "google":
            return await self._call_google(prompt)
        else:
            raise ValueError(f"Unknown model type: {self.model_config['model_type']}")

    async def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API asynchronously."""
        client = openai.AsyncOpenAI(api_key=self.openai_api_key)
        try:
            response = await client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides detailed answers to open-ended questions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.model_config["temperature"],
                max_tokens=self.model_config["max_tokens"]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling OpenAI API for model {self.model_name}: {str(e)}")
            raise

    async def _call_google(self, prompt: str) -> str:
        """Call Google Gemini API using standard REST API."""
        max_retries = 5
        base = 1.0  # Start with 1 second base delay
        
        for attempt in range(max_retries):
            try:
                # 使用标准Generative Language API
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
                headers = {"Content-Type": "application/json"}
                params = {"key": self.google_api_key}
                
                data = {
                    "systemInstruction": {
                        "parts": [{"text": "You are a helpful assistant that provides detailed answers to open-ended questions."}]
                    },
                    "contents": [{
                        "role": "user",
                        "parts": [{"text": prompt}]
                    }],
                    "generationConfig": {
                        "temperature": self.model_config["temperature"],
                        "maxOutputTokens": self.model_config["max_tokens"]
                    }
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, headers=headers, params=params, json=data) as response:
                        if response.status == 200:
                            result = await response.json()
                            return result["candidates"][0]["content"]["parts"][0]["text"]
                        else:
                            error_text = await response.text()
                            print(f"Google API error (attempt {attempt + 1}): {error_text}")
                            
                            if "quota" in error_text.lower() or "rate" in error_text.lower():
                                # Rate limit or quota exceeded
                                wait_time = base * (2 ** attempt)
                                print(f"Rate limited. Waiting {wait_time} seconds...")
                                await asyncio.sleep(wait_time)
                            else:
                                # Other error, don't retry
                                break
                                
            except Exception as e:
                print(f"Error calling Google API (attempt {attempt + 1}): {str(e)}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(base * (2 ** attempt))

    async def run_benchmark(self, output_path: str, concurrency: int = 500):
        """Run benchmark with high concurrency processing"""
        results = []
        total = len(self.test_set)
        
        print(f"\nTotal questions: {total}")
        print(f"Processing model: {self.model_name}")
        
        # 设置并发数
        print(f"Concurrency: {concurrency}")
        
        # 创建信号量控制并发
        semaphore = asyncio.Semaphore(concurrency)
        
        async def process_item_with_semaphore(item):
            """使用信号量处理单个项目"""
            async with semaphore:
                try:
                    # 检查是否可以从golden lookup获取
                    if self.setting == "mix" and self.golden_lookup:
                        question = item['question']
                        passage = item.get('passage', '')
                        lookup_key = f"{question}_{passage}"
                        
                        if lookup_key in self.golden_lookup:
                            print(f"✅ Found in golden lookup: {question[:50]}...")
                            return {
                                'question': item['question'],
                                'correct_answer': item['correct_answer'],
                                'model_answer': self.golden_lookup[lookup_key]['model_answer'],
                                'answer_coverage': None,
                                'difficulty': item['difficulty'],
                                'passage': item.get('passage', '')
                            }
                    
                    # 需要生成答案
                    print(f"🔄 Generating answer for: {item['question'][:50]}...")
                    response = await self.generate_answer(
                        query=item['question'],
                        context=item['context'],
                        difficulty=item['difficulty']
                    )
                    
                    return {
                        'question': item['question'],
                        'correct_answer': item['correct_answer'],
                        'model_answer': response,
                        'answer_coverage': None,
                        'difficulty': item['difficulty'],
                        'passage': item.get('passage', '')
                    }
                    
                except Exception as e:
                    print(f"❌ Error processing item: {str(e)}")
                    return {
                        'question': item['question'],
                        'correct_answer': item['correct_answer'],
                        'model_answer': '',
                        'answer_coverage': None,
                        'difficulty': item['difficulty'],
                        'passage': item.get('passage', '')
                    }
        
        # 创建所有任务
        print(f"\nCreating {total} tasks for concurrent processing...")
        tasks = [process_item_with_semaphore(item) for item in self.test_set]
        
        # 并发执行所有任务
        print("Starting concurrent processing...")
        start_time = time.time()
        
        # 使用asyncio.gather并发执行所有任务
        processed_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        print(f"Concurrent processing completed in {end_time - start_time:.2f} seconds")
        
        # 处理结果
        for i, result in enumerate(processed_results):
            if isinstance(result, Exception):
                print(f"❌ Task {i} failed with exception: {result}")
                # 创建错误结果
                item = self.test_set[i]
                results.append({
                    'question': item['question'],
                    'correct_answer': item['correct_answer'],
                    'model_answer': '',
                    'answer_coverage': None,
                    'difficulty': item['difficulty'],
                    'passage': item.get('passage', '')
                })
            else:
                results.append(result)
        
        # 保存结果
        print("\nSaving results...")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print("Results saved successfully!")
        
        # 计算统计信息
        total_questions = len(results)
        valid_answers = sum(1 for r in results if r['model_answer'].strip())
        
        print(f"\nResults for {self.model_name} ({self.setting}):")
        print(f"Total questions: {total_questions}")
        print(f"Valid answers: {valid_answers}")
        print(f"Response rate: {(valid_answers / total_questions) * 100:.2f}%")
        print(f"Processing time: {end_time - start_time:.2f} seconds")
        print(f"Average time per question: {(end_time - start_time) / total_questions:.2f} seconds")

async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Run benchmark for closed models on OE questions with difficulty-aware generation')
    parser.add_argument('--test_mode', action='store_true', help='Run in test mode (first 10 items only)')
    parser.add_argument('--model', type=str, required=True, help='Model name (e.g., gpt-4o, gemini-2.5-pro)')
    parser.add_argument('--concurrency', type=int, help='Concurrency level (auto-determined if not specified)')
    args = parser.parse_args()

    base_dir = "/home/shared/RAG_DATA"
    # 修改为OE测试集路径，使用带difficulty的文件
    test_sets = [
        f"{base_dir}/DATA/final_OE/base_oe_with_difficulty.json",
        f"{base_dir}/DATA/final_OE/golden_oe_with_difficulty.json",
        f"{base_dir}/DATA/final_OE/mix_oe_with_difficulty.json"
    ]

    model_name = args.model
    
    if model_name not in CLOSED_MODEL_CONFIGS:
        print(f"Error: Model {model_name} not supported. Supported models: {list(CLOSED_MODEL_CONFIGS.keys())}")
        return

    # 根据模型自动设置并发数
    if args.concurrency is None:
        concurrency = 500 if model_name.startswith("gemini") else 3000  # GPT用3000，Gemini用500
        print(f"Auto-setting concurrency: {concurrency} for {model_name}")
    else:
        concurrency = args.concurrency
        print(f"Using specified concurrency: {concurrency}")

    print(f"\nProcessing model: {model_name}")
    
    for test_set in test_sets:
        setting = os.path.basename(test_set).split('_')[0]  # base/golden/mix
        # 修改输出路径为OE路径
        output_path = f"{base_dir}/DATA/local_OE/{model_name}/{setting}_oe_with_difficulty.json"
        
        if os.path.exists(output_path):
            print(f"\nOutput file for {setting} setting already exists at: {output_path}")
            print("Skipping this setting...")
            continue
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f"\nProcessing {setting} setting...")
        
        evaluator = ClosedModelOEEvaluator(test_set, model_name)
        
        if args.test_mode:
            evaluator.test_set = evaluator.test_set[:10]
            print("Running in test mode (first 10 items only)")
        
        await evaluator.run_benchmark(output_path, concurrency)

if __name__ == "__main__":
    asyncio.run(main())
