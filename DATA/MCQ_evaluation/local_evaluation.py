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

# 模型配置 - 更新为实际下载的模型
MODEL_CONFIGS = {
    "llama-3-8b": {
        "path": "/home/shared/RAG_DATA/DATA/models/llama-3-8b",
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "max_sequence_length": 8192,
        "max_new_tokens": 32,
        "generation_config": {
            # "temperature": 1e-5,  # 极低温度，接近确定性
            "do_sample": False,   # 贪婪解码
            "max_new_tokens": 32
        }
    },
    "llama-3.2-3b-instruct": {
        "path": "/home/shared/RAG_DATA/DATA/models/llama-3.2-3b-instruct",
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "max_sequence_length": 8192,
        "max_new_tokens": 32,
        "generation_config": {
            # "temperature": 1e-5,  # 极低温度，接近确定性
            "do_sample": False,   # 贪婪解码
            "max_new_tokens": 32
        }
    },
    "Mistral-7B-Instruct-v0.2": {
        "path": "/home/shared/RAG_DATA/DATA/models/Mistral-7B-Instruct-v0.2",
        "max_sequence_length": 8192,  # 32K max, 使用8K作为安全值
        "torch_dtype": torch.float16,
        "max_new_tokens": 32,  # MCQ答案通常很短
        "device_map": "auto",
        "generation_config": {
            "temperature": 1e-5,  # 极低温度，接近确定性
            "do_sample": False,   # 贪婪解码
            "max_new_tokens": 32
        }
    },
    "qwen-2.5-3b-instruct": {
        "path": "/home/shared/RAG_DATA/DATA/models/qwen-2.5-3b-instruct",
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "max_sequence_length": 8192,
        "max_new_tokens": 32,
        "generation_config": {
            # "temperature": 1e-5,
            "do_sample": False,
            "max_new_tokens": 32
        }
    },
    "qwen-3-4b": {
        "path": "/home/shared/RAG_DATA/DATA/models/qwen-3-4b",
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "max_sequence_length": 8192,
        "max_new_tokens": 32,
            "generation_config": {
            # "temperature": 1e-5,
            "do_sample": False,
            "max_new_tokens": 32
        }
    },
    "qwen-3-8b": {
        "path": "/home/shared/RAG_DATA/DATA/models/qwen-3-8b",
        "max_new_tokens": 32,  # MCQ答案通常很短
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "max_sequence_length": 8192,
        "generation_config": {
            # "temperature": 1e-5,  # 极低温度，接近确定性
            "do_sample": False,   # 贪婪解码
            "max_new_tokens": 32
        }
    },
    "deepseek-v3-7b": {
        "path": "/home/shared/RAG_DATA/DATA/models/deepseek-v3-7b",
        "max_new_tokens": 32,
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "max_sequence_length": 8192,
        "generation_config": {
            # "temperature": 1e-5,  # 极低温度，接近确定性
            "do_sample": False,    # 启用采样但温度极低
            "max_new_tokens": 32
        }
    },
    "phi-2": {
        "path": "/home/shared/RAG_DATA/DATA/models/phi-2",
        "max_new_tokens": 32,
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "max_sequence_length": 2048,  # 实际限制就是2K
        "generation_config": {
            # "temperature": 1e-5,  # 极低温度，接近确定性
            "do_sample": False,    # 启用采样但温度极低
            "max_new_tokens": 32
        }
    },
    "gemma-7b": {
        "path": "/home/shared/RAG_DATA/DATA/models/gemma-7b",
        "max_new_tokens": 32,
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "max_sequence_length": 2048,
        "generation_config": {
            # "temperature": 1e-5,  # 极低温度，接近确定性
            "do_sample": False,    # 启用采样但温度极低
            "max_new_tokens": 32
        }
    },
    "Llama-3.2-1B-Instruct": {
        "path": "/home/shared/RAG_DATA/DATA/models/Llama-3.2-1B-Instruct",
        "torch_dtype": torch.float16,
        "max_new_tokens": 32,
        "device_map": "auto",
        "max_sequence_length": 2048,
        "generation_config": {
            # "temperature": 1e-5,
            "do_sample": False,
            "max_new_tokens": 32
        }
    },
    "qwen-3-0.6b": {
        "path": "/home/shared/RAG_DATA/DATA/models/qwen-3-0.6b",
        "torch_dtype": torch.float16,
        "max_new_tokens": 32,
        "device_map": "auto",
        "max_sequence_length": 8192,  # 40K max, 使用8K作为安全值
        "generation_config": {
            # "temperature": 1e-5,
            "do_sample": False,
            "max_new_tokens": 32
        }
    },
    "Hunyuan-7B-Instruct": {
        "path": "/home/shared/RAG_DATA/DATA/models/Hunyuan-7B-Instruct",
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "trust_remote_code": True,
        "max_sequence_length": 8192,
        "max_new_tokens": 32,
        "generation_config": {
            "do_sample": False,
            "max_new_tokens": 32
        }
    },
    "Hunyuan-4B-Instruct": {
        "path": "/home/shared/RAG_DATA/DATA/models/Hunyuan-4B-Instruct",
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "trust_remote_code": True,
        "max_sequence_length": 8192,
        "max_new_tokens": 32,
        "generation_config": {
            "do_sample": False,
            "max_new_tokens": 32
        }
    },
    "Yi-6B-Chat": {
        "path": "/home/shared/RAG_DATA/DATA/models/Yi-6B-Chat",
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "max_sequence_length": 8192,
        "max_new_tokens": 32,
        "generation_config": {
            "do_sample": False,
            "max_new_tokens": 32
        }
    },

    "Hunyuan-0.5B-Instruct": {
        "path": "/home/shared/RAG_DATA/DATA/models/Hunyuan-0.5B-Instruct",
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "trust_remote_code": True,
        "max_sequence_length": 1024,
        "max_new_tokens": 32,
        "generation_config": {
            "do_sample": False,
            "max_new_tokens": 32
        }
    },
    "AceMath-1.5B-Instruct": {
        "path": "/home/shared/RAG_DATA/DATA/models/AceMath-1.5B-Instruct",
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "max_sequence_length": 2048,
        "max_new_tokens": 32,
        "generation_config": {
            "do_sample": False,
            "max_new_tokens": 32
        }
    },
    "Falcon3-1B-Instruct": {
        "path": "/home/shared/RAG_DATA/DATA/models/Falcon3-1B-Instruct",
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "max_sequence_length": 2048,
        "max_new_tokens": 32,
        "generation_config": {
            "do_sample": False,
            "max_new_tokens": 32
        }
    }
}

class LocalModelEvaluator:
    """Evaluator for local models"""
    
    def __init__(self, test_set_path: str, model_name: str, model=None, tokenizer=None):
        """Initialize evaluator"""
        self.test_set_path = test_set_path
        self.model_name = model_name
        self.model_config = MODEL_CONFIGS[model_name]
        self.setting = os.path.basename(test_set_path).split('_')[0]  # base/golden/mix
        
        # 如果是 mix 设置，加载 golden 结果作为查找表
        self.golden_lookup = {}
        if self.setting == "mix":
            self._load_golden_lookup()
        
        # 打印初始 GPU 内存状态
        print("\nInitial GPU Memory Status:")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GiB")
        print(f"Allocated Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")
        print(f"Cached Memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GiB")
        
        # Load test set
        with open(test_set_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # Convert data format
        self.test_set = []
        for item in raw_data:
            if 'multiple_choice' in item and 'gpt40' in item['multiple_choice']:
                content = item['multiple_choice']['gpt40']['content']
                passage = item.get('passage', '')  # 保存原始 passage
                processed_item = {
                    'question': content['question'],
                    'options': content['options'],  # 保留原始选项
                    'correct_answer': content['correct_option'],
                    'context': [passage] if self.setting in ['golden', 'mix'] and passage else [],  # Use passage only in golden and mix settings
                    'passage': passage  # 保存 passage 用于查找
                }
                self.test_set.append(processed_item)
        
        # Verify data format
        required_keys = ['question', 'options', 'correct_answer']
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
            self.tokenizer.padding_side = 'left'  # 设置 padding side 为 left
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_config['path'],
                torch_dtype=self.model_config['torch_dtype'],
                device_map=self.model_config['device_map'],
                trust_remote_code=True
            )
            print("Model loaded successfully!")
        else:
            self.model = model
            self.tokenizer = tokenizer
            # 设置 padding token 和 padding side
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = 'left'  # 设置 padding side 为 left
        
        # Initialize generation config - 注释掉，避免DynamicCache错误
        if self.model_config.get("generation_config") is not None:
            raw_config = self.model_config["generation_config"]
            gen_config = raw_config if isinstance(raw_config, dict) else raw_config.to_dict()
            gen_config["eos_token_id"] = self.tokenizer.eos_token_id
            gen_config["pad_token_id"] = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            self.model_config["generation_config"] = GenerationConfig(**gen_config)
        
        # Set random seed for reproducibility
        # random.seed(42)
        torch.cuda.empty_cache()  # Clean up GPU memory

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
                        'correct_answer': item['correct_answer']
                    }
                
                print(f"Loaded {len(self.golden_lookup)} golden results for lookup")
            else:
                print(f"Warning: Golden results not found at {golden_path}")
                print("Will process mix without optimization")
                
        except Exception as e:
            print(f"Error loading golden lookup: {e}")
            print("Will process mix without optimization")
    
    def _truncate_context_for_small_model(self, context: List[str], max_tokens_per_passage: int = 200) -> List[str]:
        """截断context以适应小模型的token限制"""
        if not self.model_config.get('is_small_model', False):
            return context
        
        truncated_context = []
        for passage in context:
            # 简单按字符数截断，大约4个字符=1个token
            if len(passage) > max_tokens_per_passage * 4:
                passage = passage[:max_tokens_per_passage * 4] + "..."
            truncated_context.append(passage)
        
        return truncated_context

    def generate_answer(self, query: str, context: List[str], options: List[str]) -> str:
        """Generate answer for a question"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # 为小模型截断context
                context = self._truncate_context_for_small_model(context)
                
                # Format options as A, B, C, D
                options_str = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
                context_str = "\n".join(context) if context else ""
                
                # 检查是否为小模型
                is_small_model = self.model_config.get('is_small_model', False)
                
                # Build prompt based on setting and model size
                # 统一为0-shot prompt
                if context_str:
                    prompt = f"""Passage: {context_str}\n\nQuestion: {query}\nOptions:\n{options_str}\n\nAnswer:"""
                else:
                    prompt = f"""Question: {query}\nOptions:\n{options_str}\n\nAnswer:"""
                
                print("\nGenerating response...")
                print(f"Prompt length: {len(prompt)}")
                
                # Generate response
                print("Tokenizing input...")
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.model_config['max_sequence_length']
                ).to(self.model.device)
                
                # 移除token_type_ids以避免与某些模型的兼容性问题
                if 'token_type_ids' in inputs:
                    del inputs['token_type_ids']
                
                print(f"Input length: {len(inputs['input_ids'][0])}")
                print("Starting generation...")
                
                # Generate with model-specific parameters
                with torch.no_grad():
                    try:
                        # Add progress indicator
                        print("Generating tokens...", end="", flush=True)
                        start_time = time.time()
                        # 使用更简单的生成参数，避免版本兼容性问题
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=self.model_config["max_new_tokens"],
                            do_sample=False,
                            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                            eos_token_id=self.tokenizer.eos_token_id
                        )
                        end_time = time.time()
                        print(f"\nGeneration completed in {end_time - start_time:.2f} seconds!")
                    except Exception as e:
                        print(f"Error during generation: {str(e)}")
                        raise
                
                print(f"Output length: {len(outputs[0])}")
                
                # Decode and return response
                print("Decoding response...")
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Remove the prompt from the response
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
                time.sleep(1)  # Wait for 1 second before retrying

    def parse_answer(self, response: str) -> Dict[str, str]:
        """Parse model's answer from response"""
        try:
            # 首先尝试匹配 "Answer: X" 格式
            answer_match = re.search(r'Answer:\s*([A-D])', response)
            if answer_match:
                answer = answer_match.group(1)
            else:
                # 如果没有找到标准格式，尝试直接匹配 A/B/C/D
                answer_match = re.search(r'[A-D]', response)
                if answer_match:
                    answer = answer_match.group(0)
                else:
                    # 如果还是没有找到，尝试匹配选项前缀
                    answer_match = re.search(r'([A-D])\.', response)
                    if answer_match:
                        answer = answer_match.group(1)
                    else:
                        raise ValueError("No valid answer found in response")
            
            # 尝试匹配解释
            explanation_match = re.search(r'Explanation:\s*(.*?)(?=\n|$)', response, re.DOTALL)
            explanation = explanation_match.group(1).strip() if explanation_match else ""
            
            return {
                'answer': answer,
                'explanation': explanation
            }
        except Exception as e:
            print(f"Error parsing answer: {str(e)}")
            print(f"Response: {response}")
            return {
                'answer': '',
                'explanation': str(e)
            }

    def run_benchmark(self, output_path: str):
        """Run benchmark on test set"""
        results = []
        total = len(self.test_set)
        
        # 根据设置类型调整批处理大小
        if self.setting == "base":
            batch_size = 16
        else:  # mix 和 golden 设置使用更小的批处理大小
            batch_size = 8 # 改为4，因为每个prompt会更大
        
        print(f"\nTotal questions: {total}")
        print(f"Batch size: {batch_size}")
        print(f"Number of batches: {(total + batch_size - 1) // batch_size}\n")
        
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
                # 对于 mix 设置，过滤掉已经在 golden 查找表中的问题
                if self.setting == "mix":
                    filtered_batch = []
                    for item in batch:
                        question = item['question']
                        passage = item.get('passage', '')
                        lookup_key = f"{question}_{passage}"
                        if lookup_key not in self.golden_lookup:
                            filtered_batch.append(item)
                    
                    if filtered_batch:
                        # 准备批次数据（只包含需要模型推理的问题）
                        batch_queries = [item['question'] for item in filtered_batch]
                        batch_contexts = [item['context'] for item in filtered_batch]
                        batch_options = [item['options'] for item in filtered_batch]
                        
                        # 生成答案
                        batch_responses = self.generate_answer_batch(
                            queries=batch_queries,
                            contexts=batch_contexts,
                            options_list=batch_options
                        )
                    else:
                        # 所有问题都在 golden 查找表中，不需要模型推理
                        batch_responses = []
                        filtered_batch = []
                else:
                    # 非 mix 设置，正常处理
                    filtered_batch = batch
                    batch_queries = [item['question'] for item in batch]
                    batch_contexts = [item['context'] for item in batch]
                    batch_options = [item['options'] for item in batch]
                    
                    # 生成答案
                    batch_responses = self.generate_answer_batch(
                        queries=batch_queries,
                        contexts=batch_contexts,
                        options_list=batch_options
                    )
                
                # 处理每个答案（包括从 golden 查找表获取的和模型生成的）
                processed_items = set()
                
                # 处理模型生成的结果
                for item, response in zip(filtered_batch, batch_responses):
                    try:
                        # 检查是否可以从 golden 查找表中获取结果
                        question = item['question']
                        passage = item.get('passage', '')
                        lookup_key = f"{question}_{passage}"
                        
                        if self.setting == "mix" and lookup_key in self.golden_lookup:
                            # 直接使用 golden 结果
                            golden_result = self.golden_lookup[lookup_key]
                            result = {
                                'question': item['question'],
                                'options': item['options'],
                                'correct_answer': item['correct_answer'],
                                'model_answer': golden_result['model_answer'],
                                'model_explanation': golden_result['model_explanation'],
                                'is_correct': golden_result['is_correct']
                            }
                            
                            # 确保passage字段正确传递到结果中
                            if 'passage' in item:
                                result['passage'] = item['passage']
                                
                            print(f"✅ Found in golden lookup: {item['question'][:50]}...")
                        else:
                            # 解析模型生成的答案
                            parsed_answer = self.parse_answer(response)
                            
                            # 验证答案
                            is_correct = parsed_answer['answer'] == item['correct_answer']
                            
                            # 保存结果
                            result = {
                                'question': item['question'],
                                'options': item['options'],
                                'correct_answer': item['correct_answer'],
                                'model_answer': parsed_answer['answer'],
                                'model_explanation': parsed_answer['explanation'],
                                'is_correct': is_correct
                            }
                            
                            # 确保passage字段正确传递到结果中
                            if 'passage' in item:
                                result['passage'] = item['passage']
                        
                        results.append(result)
                        
                        # 打印进度
                        if self.setting == "mix" and lookup_key in self.golden_lookup:
                            print(f"Question: {item['question'][:100]}... (from golden)")
                            print(f"Correct answer: {item['correct_answer']}")
                            print(f"Model answer: {result['model_answer']}")
                            print(f"Correct: {result['is_correct']}\n")
                        else:
                            print(f"Question: {item['question'][:100]}...")
                            print(f"Correct answer: {item['correct_answer']}")
                            print(f"Model answer: {result['model_answer']}")
                            print(f"Correct: {result['is_correct']}\n")
                        
                        processed_items.add(lookup_key)
                        
                    except Exception as e:
                        print(f"Error processing answer: {str(e)}")
                        error_result = {
                            'question': item['question'],
                            'options': item['options'],
                            'correct_answer': item['correct_answer'],
                            'model_answer': '',
                            'model_explanation': str(e),
                            'is_correct': False
                        }
                        
                        # 确保passage字段正确传递到结果中
                        if 'passage' in item:
                            error_result['passage'] = item['passage']
                            
                        results.append(error_result)
                        processed_items.add(lookup_key)
                
                # 处理在 golden 查找表中但不在 filtered_batch 中的问题
                if self.setting == "mix":
                    for item in batch:
                        question = item['question']
                        passage = item.get('passage', '')
                        lookup_key = f"{question}_{passage}"
                        if lookup_key not in processed_items and lookup_key in self.golden_lookup:
                            golden_result = self.golden_lookup[lookup_key]
                            result = {
                                'question': item['question'],
                                'options': item['options'],
                                'correct_answer': item['correct_answer'],
                                'model_answer': golden_result['model_answer'],
                                'model_explanation': golden_result['model_explanation'],
                                'is_correct': golden_result['is_correct']
                            }
                            
                            # 确保passage字段正确传递到结果中
                            if 'passage' in item:
                                result['passage'] = item['passage']
                                
                            results.append(result)
                            print(f"✅ Added from golden lookup: {item['question'][:50]}...")
                
                # 清理缓存
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error processing batch: {str(e)}")
                # 如果批处理失败，回退到单个处理
                for item in batch:
                    try:
                        response = self.generate_answer(
                            query=item['question'],
                            context=item['context'],
                            options=item['options']
                        )
                        parsed_answer = self.parse_answer(response)
                        is_correct = parsed_answer['answer'] == item['correct_answer']
                        single_result = {
                            'question': item['question'],
                            'options': item['options'],
                            'correct_answer': item['correct_answer'],
                            'model_answer': parsed_answer['answer'],
                            'model_explanation': parsed_answer['explanation'],
                            'is_correct': is_correct
                        }
                        
                        # 确保passage字段正确传递到结果中
                        if 'passage' in item:
                            single_result['passage'] = item['passage']
                            
                        results.append(single_result)
                    except Exception as e:
                        print(f"Error processing single item: {str(e)}")
                        error_result = {
                            'question': item['question'],
                            'options': item['options'],
                            'correct_answer': item['correct_answer'],
                            'model_answer': '',
                            'model_explanation': str(e),
                            'is_correct': False
                        }
                        
                        # 确保passage字段正确传递到结果中
                        if 'passage' in item:
                            error_result['passage'] = item['passage']
                            
                        results.append(error_result)
                
                # 清理缓存
                torch.cuda.empty_cache()
        
        # 保存结果
        print("\nSaving results...")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print("Results saved successfully!")
        
        # 计算统计信息
        total_questions = len(results)
        correct_answers = sum(1 for r in results if r['is_correct'])
        accuracy = (correct_answers / total_questions) * 100 if total_questions > 0 else 0
        
        print(f"\nResults for {self.model_name} ({self.setting}):")
        print(f"Total questions: {total_questions}")
        print(f"Correct answers: {correct_answers}")
        print(f"Accuracy: {accuracy:.2f}%")

    def generate_answer_batch(self, queries: List[str], contexts: List[List[str]], options_list: List[List[str]]) -> List[str]:
        """Generate answers for a batch of questions"""
        try:
            # 检查是否为小模型
            is_small_model = self.model_config.get('is_small_model', False)
            
            # 准备批次提示词
            batch_prompts = []
            for query, context, options in zip(queries, contexts, options_list):
                # 为小模型截断context
                context = self._truncate_context_for_small_model(context)
                # 移除选项前缀
                clean_options = [opt.split('. ', 1)[1] for opt in options]
                options_str = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(clean_options)])
                context_str = "\n".join(context) if context else ""
                # 统一为0-shot prompt
                if context_str:
                    prompt = f"""Passage: {context_str}\n\nQuestion: {query}\nOptions:\n{options_str}\n\nAnswer:"""
                else:
                    prompt = f"""Question: {query}\nOptions:\n{options_str}\n\nAnswer:"""
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
            
            # 移除token_type_ids以避免与某些模型的兼容性问题
            if 'token_type_ids' in inputs:
                del inputs['token_type_ids']
            
            print(f"Batch input shape: {inputs['input_ids'].shape}")
            
            # 生成回答
            print("\nGenerating batch responses...")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.model_config["max_new_tokens"],
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
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
    parser = argparse.ArgumentParser(description='Run benchmark for local models')
    parser.add_argument('--test_mode', action='store_true', help='Run in test mode (first 10 items only)')
    args = parser.parse_args()

    base_dir = "/home/shared/RAG_DATA"
    test_sets = [
        f"{base_dir}/DATA/final_mcq/base_2000.json",
        f"{base_dir}/DATA/final_mcq/golden_2000.json",  # golden一般还是用原始的
        f"{base_dir}/DATA/final_mcq/mix_2000.json"
    ]

    # 你要评测的所有模型
    model_names = list(MODEL_CONFIGS.keys())

    for model_name in model_names:
        print(f"\nProcessing model: {model_name}")
        print(f"Loading model {model_name} from {MODEL_CONFIGS[model_name]['path']}...")
        
        try:
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
            
        except Exception as e:
            print(f"Error loading model {model_name}: {str(e)}")
            print(f"Failed to load model {model_name}, skipping...")
            continue

        for test_set in test_sets:
            setting = os.path.basename(test_set).split('_')[0]  # base/golden/mix
            output_path = f"{base_dir}/DATA/local_MCQ/{model_name}/{setting}_test.json"
            if os.path.exists(output_path):
                print(f"\nOutput file for {setting} setting already exists at: {output_path}")
                print("Skipping this setting...")
                continue
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            print(f"\nProcessing {setting} setting...")
            evaluator = LocalModelEvaluator(test_set, model_name, model, tokenizer)
            if args.test_mode:
                evaluator.test_set = evaluator.test_set[:10]
                print("Running in test mode (first 10 items only)")
            evaluator.run_benchmark(output_path)

if __name__ == "__main__":
    main() 