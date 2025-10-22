import json
import os
import argparse
from typing import List, Dict, Any
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import time
import re
from dotenv import load_dotenv
import random

# 模型配置
MODEL_CONFIGS = {
    "llama-3-8b": {
        "path": "/home/shared/RAG_DATA/benchmark/models/llama-3-8b",
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "max_sequence_length": 8192,
        "max_new_tokens": 512,  # 增加token数量用于简答题
        "do_sample": True,
        "generation_config": {
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 50,
            "repetition_penalty": 1.2,
            "no_repeat_ngram_size": 3,
            "do_sample": True,
            "max_new_tokens": 512
        }
    },
    "mistral-3-7b": {
        "path": "/home/shared/RAG_DATA/benchmark/models/mistral-3-7b",
        "max_sequence_length": 4096,
        "do_sample": True,
        "torch_dtype": torch.float16,
        "max_new_tokens": 512,
        "device_map": "auto",
        "generation_config": {
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 50,
            "repetition_penalty": 1.2,
            "no_repeat_ngram_size": 3,
            "do_sample": True,
            "max_new_tokens": 512
        }
    },
    "qwen-3-8b": {
        "path": "/home/shared/RAG_DATA/benchmark/models/qwen-3-8b",
        "max_new_tokens": 512,
        "temperature": 0.1,
        "do_sample": True,
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "max_sequence_length": 8192,
        "do_sample": True,
        "generation_config": {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "do_sample": True,
            "repetition_penalty": 1.15,
            "no_repeat_ngram_size": 3,
            "max_new_tokens": 512
        }
    },
    "deepseek-v3-7b": {
        "path": "/home/shared/RAG_DATA/benchmark/models/deepseek-v3-7b",
        "max_new_tokens": 512,
        "do_sample": True,
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "max_sequence_length": 8192,
        "generation_config": {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "do_sample": True,
            "repetition_penalty": 1.1,
            "no_repeat_ngram_size": 3,
            "do_sample": True,
            "max_new_tokens": 512
        }
    },
    "phi-2": {
        "path": "/home/shared/RAG_DATA/benchmark/models/phi-2",
        "max_new_tokens": 512,
        "temperature": 0.1,
        "do_sample": True,
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "max_sequence_length": 2048,
        "generation_config": {
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.2,
            "length_penalty": 1.0,
            "no_repeat_ngram_size": 3,
            "max_new_tokens": 512
        }
    },
    "gemma-7b": {
        "path": "/home/shared/RAG_DATA/benchmark/models/gemma-7b",
        "max_new_tokens": 512,
        "torch_dtype": "float16",
        "device_map": "auto",
        "max_sequence_length": 2048,
        "do_sample": True,
        "generation_config": {
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.2,
            "length_penalty": 1.0,
            "no_repeat_ngram_size": 3,
            "max_new_tokens": 512
        }
    },
    "TinyLlama": {
        "path": "/home/shared/RAG_DATA/benchmark/models/TinyLlama",
        "max_new_tokens": 512,
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "max_sequence_length": 2048,
        "is_small_model": True,
        "generation_config": {
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.2,
            "length_penalty": 1.0,
            "no_repeat_ngram_size": 3,
            "max_new_tokens": 512
        }
    },
    "Hunyuan-7B-Instruct": {
        "path": "/home/shared/RAG_DATA/DATA/models/Hunyuan-7B-Instruct",
        "max_new_tokens": 512,
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "max_sequence_length": 4096,
        "generation_config": {
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "no_repeat_ngram_size": 3,
            "max_new_tokens": 512
        }
    },
    "Hunyuan-4B-Instruct": {
        "path": "/home/shared/RAG_DATA/DATA/models/Hunyuan-4B-Instruct",
        "max_new_tokens": 512,
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "max_sequence_length": 4096,
        "is_small_model": True,
        "generation_config": {
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "no_repeat_ngram_size": 3,
            "max_new_tokens": 512
        }
    }
}

class LocalModelEvaluator:
    """Evaluator for local models on Open-Ended questions"""
    
    def __init__(self, test_set_path: str, model_name: str, model=None, tokenizer=None):
        """Initialize evaluator"""
        self.test_set_path = test_set_path
        self.model_name = model_name
        self.model_config = MODEL_CONFIGS[model_name]
        self.setting = os.path.basename(test_set_path).split('_')[2]  # base/golden/mix
        
        # 打印初始 GPU 内存状态
        print("\nInitial GPU Memory Status:")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GiB")
        print(f"Allocated Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")
        print(f"Cached Memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GiB")
        
        # Load test set
        with open(test_set_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # Convert data format for OE questions
        self.test_set = []
        for item in raw_data:
            if 'open_ended' in item and 'gpt40' in item['open_ended']:
                content = item['open_ended']['gpt40']['content']
                processed_item = {
                    'original_query': item.get('original_query', ''),
                    'question': content['question'],
                    'context': [item['passage']] if self.setting in ['golden', 'mix'] and 'passage' in item else []
                }
                self.test_set.append(processed_item)
        
        # Verify data format
        required_keys = ['question']
        for i, item in enumerate(self.test_set):
            missing_keys = [key for key in required_keys if key not in item]
            if missing_keys:
                print(f"Warning: Item {i} is missing keys: {missing_keys}")
        
        # Load model and tokenizer if not provided
        if model is None or tokenizer is None:
            print(f"Loading model {model_name} from {self.model_config['path']}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_config['path'],
                trust_remote_code=True
            )
            # 设置 padding token 和 padding side
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = 'left'
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_config['path'],
                torch_dtype=self.model_config['torch_dtype'],
                device_map=self.model_config['device_map'],
                trust_remote_code=True
            )
        else:
            self.model = model
            self.tokenizer = tokenizer
            # 确保tokenizer设置正确
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = 'left'
        
        print("Model loaded successfully!")
        
        # 打印最终 GPU 内存状态
        print("\nFinal GPU Memory Status:")
        print(f"Allocated Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")
        print(f"Cached Memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GiB")

    def _truncate_context_for_small_model(self, context: List[str], max_tokens_per_passage: int = 200) -> List[str]:
        """Truncate context for small models"""
        if not context:
            return context
        
        truncated_context = []
        for passage in context:
            tokens = self.tokenizer.encode(passage)
            if len(tokens) > max_tokens_per_passage:
                truncated_tokens = tokens[:max_tokens_per_passage]
                truncated_passage = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
                truncated_context.append(truncated_passage)
            else:
                truncated_context.append(passage)
        
        return truncated_context

    def generate_answer(self, query: str, context: List[str]) -> str:
        """Generate answer for a single question"""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # 检查是否为小模型
                is_small_model = self.model_config.get('is_small_model', False)
                
                # 为小模型截断context
                context = self._truncate_context_for_small_model(context)
                
                # 构建prompt
                if not context:  # base setting
                    if is_small_model:
                        prompt = f"""Answer the following question in detail and provide explanation:

Question: {query}

Answer:"""
                    else:
                        prompt = f"""You are a helpful assistant that provides detailed answers to open-ended questions. Please provide a comprehensive answer that addresses all aspects of the question, followed by a brief explanation of your reasoning.

Question: {query}

Answer:"""
                else:  # mix and golden settings
                    context_str = "\n".join(context)
                    if is_small_model:
                        prompt = f"""Based on the passage, answer the question and provide explanation:

Passage: {context_str}

Question: {query}

Answer:"""
                    else:
                        prompt = f"""You are a helpful assistant that provides detailed answers to open-ended questions based on the provided passage. Please provide a comprehensive answer that addresses all aspects of the question using information from the passage, followed by a brief explanation of your reasoning.

Passage:
{context_str}

Question: {query}

Answer:"""
                
                # Tokenize input
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.model_config['max_sequence_length']
                ).to(self.model.device)
                
                # Generate response
                with torch.no_grad():
                    generation_config = GenerationConfig(**self.model_config["generation_config"])
                    outputs = self.model.generate(
                        **inputs,
                        generation_config=generation_config
                    )
                
                # Decode response
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = response[len(prompt):].strip()
                
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
                time.sleep(1)

    def run_benchmark(self, output_path: str):
        """Run benchmark on test set"""
        results = []
        total = len(self.test_set)
        
        # 根据设置类型调整批处理大小
        if self.setting == "base":
            batch_size = 16  # 简答题需要更多token，减少批处理大小
        else:  # mix 和 golden 设置使用更小的批处理大小
            batch_size = 16
        
        print(f"\nTotal questions: {total}")
        print(f"Batch size: {batch_size}")
        print(f"Number of batches: {(total + batch_size - 1) // batch_size}\n")
        
        # 如果是mix setting，检查是否有与golden一致的case
        golden_results = None
        if self.setting == "mix":
            golden_output_path = output_path.replace("mix_test.json", "golden_test.json")
            if os.path.exists(golden_output_path):
                print(f"Loading golden results from {golden_output_path} for comparison...")
                with open(golden_output_path, 'r', encoding='utf-8') as f:
                    golden_results = json.load(f)
                print(f"Loaded {len(golden_results)} golden results")
        
        # 按批次处理
        for i in range(0, total, batch_size):
            batch = self.test_set[i:i + batch_size]
            current_batch = i // batch_size + 1
            total_batches = (total + batch_size - 1) // batch_size
            print(f"\nProcessing batch {current_batch}/{total_batches} (questions {i+1}-{min(i+batch_size, total)}/{total})...")
            
            # 打印当前 GPU 内存状态
            print("\nCurrent GPU Memory Status:")
            print(f"Allocated Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")
            print(f"Cached Memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GiB")
            
            try:
                # 准备批次数据
                batch_queries = [item['question'] for item in batch]
                batch_contexts = [item['context'] for item in batch]
                
                # 如果是mix setting且有golden结果，检查哪些可以复制
                batch_responses = []
                if self.setting == "mix" and golden_results:
                    for j, (query, context) in enumerate(zip(batch_queries, batch_contexts)):
                        # 查找对应的golden结果
                        golden_match = None
                        for golden_result in golden_results:
                            if (golden_result['question'] == query and 
                                golden_result['passage'] == (context[0] if context else "")):
                                golden_match = golden_result
                                break
                        
                        if golden_match:
                            # 直接复制golden的答案
                            print(f"Mix case identical to golden for query: {query[:50]}..., copying answer")
                            batch_responses.append(golden_match['model_answer'])
                        else:
                            # 需要重新生成
                            batch_responses.append(None)  # 标记需要生成
                else:
                    # 其他情况正常生成
                    batch_responses = [None] * len(batch_queries)
                
                # 生成需要重新生成的答案
                if any(response is None for response in batch_responses):
                    # 找出需要生成的索引
                    indices_to_generate = [j for j, response in enumerate(batch_responses) if response is None]
                    queries_to_generate = [batch_queries[j] for j in indices_to_generate]
                    contexts_to_generate = [batch_contexts[j] for j in indices_to_generate]
                    
                    # 生成答案
                    generated_responses = self.generate_answer_batch(
                        queries=queries_to_generate,
                        contexts=contexts_to_generate
                    )
                    
                    # 填充结果
                    for idx, response in zip(indices_to_generate, generated_responses):
                        batch_responses[idx] = response
                
                # 处理每个答案
                for item, response in zip(batch, batch_responses):
                    try:
                        # 解析答案和解释
                        answer_parts = response.split('\n\n', 1)
                        model_answer = answer_parts[0].replace('Answer:', '').strip() if answer_parts else response
                        explanation = answer_parts[1].replace('Explanation:', '').strip() if len(answer_parts) > 1 else ""
                        
                        # 保存结果
                        result = {
                            'original_query': item.get('original_query', ''),
                            'question': item['question'],
                            'model_answer': model_answer,
                            'explanation': explanation,
                            'passage': item['context'][0] if item['context'] else ""
                        }
                        results.append(result)
                        
                        # 打印进度
                        print(f"Question: {item['question'][:100]}...")
                        print(f"Model answer: {model_answer[:200]}...")
                        print(f"Explanation: {explanation[:200]}...\n")
                        
                    except Exception as e:
                        print(f"Error processing answer: {str(e)}")
                        results.append({
                            'original_query': item.get('original_query', ''),
                            'question': item['question'],
                            'model_answer': '',
                            'explanation': '',
                            'passage': item['context'][0] if item['context'] else ""
                        })
                
                # 清理缓存
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error processing batch: {str(e)}")
                # 如果批处理失败，回退到单个处理
                for item in batch:
                    try:
                        response = self.generate_answer(
                            query=item['question'],
                            context=item['context']
                        )
                        # 解析答案和推理
                        answer_parts = response.split('\n\n', 1)
                        model_answer = answer_parts[0].replace('Answer:', '').strip() if answer_parts else response
                        reasoning = answer_parts[1].replace('Reasoning:', '').strip() if len(answer_parts) > 1 else ""
                        
                        results.append({
                            'original_query': item.get('original_query', ''),
                            'question': item['question'],
                            'model_answer': '',
                            'explanation': '',
                            'passage': item['context'][0] if item['context'] else ""
                        })
                    except Exception as e:
                        print(f"Error processing single item: {str(e)}")
                        results.append({
                            'original_query': item.get('original_query', ''),
                            'question': item['question'],
                            'model_answer': '',
                            'explanation': '',
                            'passage': item['context'][0] if item['context'] else ""
                        })
                
                # 清理缓存
                torch.cuda.empty_cache()
        
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

    def generate_answer_batch(self, queries: List[str], contexts: List[List[str]]) -> List[str]:
        """Generate answers for a batch of questions"""
        try:
            # 检查是否为小模型
            is_small_model = self.model_config.get('is_small_model', False)
            
            # 准备批次提示词
            batch_prompts = []
            for query, context in zip(queries, contexts):
                # 为小模型截断context
                context = self._truncate_context_for_small_model(context)
                
                context_str = "\n".join(context) if context else ""
                
                # 构建prompt
                if not context:  # base setting
                    if is_small_model:
                        prompt = f"""Answer the following question in detail and provide reasoning:

Question: {query}

Answer:"""
                    else:
                        prompt = f"""You are a helpful assistant that provides detailed answers to open-ended questions. Please provide a comprehensive answer that addresses all aspects of the question, followed by reasoning for your response.

Question: {query}

Answer:"""
                else:  # mix and golden settings
                    if is_small_model:
                        prompt = f"""Based on the passage, answer the question and provide reasoning:

Passage: {context_str}

Question: {query}

Answer:"""
                    else:
                        prompt = f"""You are a helpful assistant that provides detailed answers to open-ended questions based on the provided passage. Please provide a comprehensive answer that addresses all aspects of the question using information from the passage, followed by reasoning for your response.

Passage:
{context_str}

Question: {query}

Answer:"""
                batch_prompts.append(prompt)
            
            # 准备输入
            print("\nTokenizing batch input...")
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.model_config['max_sequence_length']
            ).to(self.model.device)
            
            print(f"Batch input shape: {inputs['input_ids'].shape}")
            
            # 生成回答
            print("\nGenerating batch responses...")
            with torch.no_grad():
                generation_config = GenerationConfig(**self.model_config["generation_config"])
                outputs = self.model.generate(
                    **inputs,
                    generation_config=generation_config
                )
            
            # 解码输出
            print("Decoding batch responses...")
            responses = []
            for i, output in enumerate(outputs):
                response = self.tokenizer.decode(output, skip_special_tokens=True)
                # 移除提示词
                response = response[len(batch_prompts[i]):].strip()
                responses.append(response)
            
            return responses
            
        except Exception as e:
            print(f"Error in batch generation: {str(e)}")
            raise

def main():
    """Main function"""
    # 设置 PyTorch 内存分配配置
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    parser = argparse.ArgumentParser(description='Run benchmark for local models on Open-Ended questions')
    parser.add_argument('--test_mode', action='store_true', help='Run in test mode (first 10 items only)')
    args = parser.parse_args()
    
    # Get all test sets
    base_dir = "/home/shared/RAG_DATA"
    test_sets = [
        f"{base_dir}/benchmark/OE/generated_test_sets/test_set_base_simple.json",
        f"{base_dir}/benchmark/OE/generated_test_sets/test_set_golden_simple.json",
        f"{base_dir}/benchmark/OE/generated_test_sets/test_set_mix_simple.json"
    ]
    
    # Specify the model to run
    model_name = 'gemma-7b'  # Modify model name here
    
    print(f"\nProcessing model: {model_name}")
    
    # Load model
    print(f"Loading model {model_name} from {MODEL_CONFIGS[model_name]['path']}...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CONFIGS[model_name]['path'],
        trust_remote_code=True 
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_CONFIGS[model_name]['path'],
        torch_dtype=MODEL_CONFIGS[model_name]['torch_dtype'],
        device_map=MODEL_CONFIGS[model_name]['device_map'],
        trust_remote_code=True
    )
    print("Model loaded successfully!")
    
    # Process each test set
    for test_set in test_sets:
        setting = os.path.basename(test_set).split('_')[2]  # base/golden/mix
        output_path = f"{base_dir}/benchmark/testOE_set/{model_name}/{setting}_test.json"
        
        # 检查输出文件是否已存在
        if os.path.exists(output_path):
            print(f"\nOutput file for {setting} setting already exists at: {output_path}")
            print("Skipping this setting...")
            continue
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        print(f"\nProcessing {setting} setting...")
        
        # Create evaluator
        evaluator = LocalModelEvaluator(test_set, model_name, model, tokenizer)
        
        # If test mode, only take the first 10 questions
        if args.test_mode:
            evaluator.test_set = evaluator.test_set[:10]
            print("Running in test mode (first 10 items only)")
        
        # Run evaluation
        evaluator.run_benchmark(output_path)

if __name__ == "__main__":
    main()
