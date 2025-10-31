import os
import json
import argparse
from typing import List, Dict, Any
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import time
import re
from dotenv import load_dotenv
import random

# Model configurations
MODEL_CONFIGS = {
    "llama-3-8b": {
        "path": "/home/shared/RAG_DATA/benchmark/models/llama-3-8b",
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "max_sequence_length": 8192,
        "generation_config": {
            "temperature": 1e-5,
            "do_sample": False,
            "max_new_tokens": 64
        }
    },
    "mistral-3-7b": {
        "path": "/home/shared/RAG_DATA/benchmark/models/mistral-3-7b",
        "max_sequence_length": 4096,
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "generation_config": {
            "temperature": 1e-5,
            "do_sample": False,
            "max_new_tokens": 64
        }
    },
    "mistral-8b": {
        "path": "/home/shared/RAG_DATA/benchmark/models/mistral-8b",
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "max_sequence_length": 8192,
        "generation_config": {
            "temperature": 1e-5,
            "do_sample": False,
            "max_new_tokens": 64
        }
    },
    "qwen-2.5-3b-instruct": {
        "path": "/home/shared/RAG_DATA/benchmark/models/qwen-2.5-3b-instruct",
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "max_sequence_length": 8192,
        "generation_config": {
            "temperature": 1e-5,
            "do_sample": False,
            "max_new_tokens": 64
        }
    },
    "qwen-2.5-7b-instruct": {
        "path": "/home/shared/RAG_DATA/benchmark/models/qwen-2.5-7b-instruct",
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "max_sequence_length": 8192,
        "generation_config": {
            "temperature": 1e-5,
            "do_sample": False,
            "max_new_tokens": 64
        }
    },
    "qwen-3-4b": {
        "path": "/home/shared/RAG_DATA/benchmark/models/qwen-3-4b",
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "max_sequence_length": 8192,
        "generation_config": {
            "temperature": 1e-5,
            "do_sample": False,
            "max_new_tokens": 64
        }
    },
    "qwen-3-8b": {
        "path": "/home/shared/RAG_DATA/benchmark/models/qwen-3-8b",
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "max_sequence_length": 8192,
        "generation_config": {
            "temperature": 1e-5,
            "do_sample": False,
            "max_new_tokens": 64
        }
    },
    "deepseek-v3-7b": {
        "path": "/home/shared/RAG_DATA/benchmark/models/deepseek-v3-7b",
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "max_sequence_length": 8192,
        "generation_config": {
            "temperature": 1e-5,
            "do_sample": False,
            "max_new_tokens": 64
        }
    },
    "phi-2": {
        "path": "/home/shared/RAG_DATA/benchmark/models/phi-2",
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "max_sequence_length": 2048,
        "generation_config": {
            "temperature": 1e-5,
            "do_sample": False,
            "max_new_tokens": 64
        }
    },
    "phi-4": {
        "path": "/home/shared/RAG_DATA/benchmark/models/phi-4",
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "max_sequence_length": 2048,
        "generation_config": {
            "temperature": 1e-5,
            "do_sample": False,
            "max_new_tokens": 64
        }
    },
    "gemma-7b": {
        "path": "/home/shared/RAG_DATA/benchmark/models/gemma-7b",
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "max_sequence_length": 2048,
        "generation_config": {
            "temperature": 1e-5,
            "do_sample": False,
            "max_new_tokens": 64
        }
    },
    "gemma-2-2b": {
        "path": "/home/shared/RAG_DATA/benchmark/models/gemma-2-2b",
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "max_sequence_length": 2048,
        "generation_config": {
            "temperature": 1e-5,
            "do_sample": False,
            "max_new_tokens": 64
        }
    },
    "TinyLlama": {
        "path": "/home/shared/RAG_DATA/benchmark/models/TinyLlama",
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "max_sequence_length": 2048,
        "is_small_model": True,
        "generation_config": {
            "temperature": 1e-5,
            "do_sample": False,
            "max_new_tokens": 64
        }
    }
}


class LocalModelEvaluator:
    """Evaluator for local MCQ benchmark models."""

    def __init__(self, test_set_path: str, model_name: str, model=None, tokenizer=None):
        self.test_set_path = test_set_path
        self.model_name = model_name
        self.model_config = MODEL_CONFIGS[model_name]
        self.setting = os.path.basename(test_set_path).split('_')[2]  # base/golden/mix

        self.golden_lookup = {}
        if self.setting == "mix":
            self._load_golden_lookup()

        print("\nInitial GPU Memory Status:")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GiB")
        print(f"Allocated Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")
        print(f"Cached Memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GiB")

        # Load test set
        with open(test_set_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        self.test_set = []
        for item in raw_data:
            if 'multiple_choice' in item and 'gpt40' in item['multiple_choice']:
                content = item['multiple_choice']['gpt40']['content']
                passage = item.get('passage', '')
                self.test_set.append({
                    'question': content['question'],
                    'options': content['options'],
                    'correct_answer': content['correct_option'],
                    'context': [passage] if self.setting in ['golden', 'mix'] and passage else [],
                    'passage': passage
                })

        if model is None or tokenizer is None:
            print(f"Loading model {model_name} from {self.model_config['path']}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_config['path'], trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = 'left'

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_config['path'],
                torch_dtype=self.model_config['torch_dtype'],
                device_map=self.model_config['device_map'],
                trust_remote_code=True
            )
            print("Model loaded successfully.")
        else:
            self.model = model
            self.tokenizer = tokenizer
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = 'left'

        if self.model_config.get("generation_config") is not None:
            raw_config = self.model_config["generation_config"]
            gen_config = raw_config if isinstance(raw_config, dict) else raw_config.to_dict()
            gen_config["eos_token_id"] = self.tokenizer.eos_token_id
            gen_config["pad_token_id"] = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            self.model_config["generation_config"] = GenerationConfig(**gen_config)

        torch.cuda.empty_cache()

    def _load_golden_lookup(self):
        """Load golden results for mix lookup optimization."""
        base_dir = "/home/shared/RAG_DATA"
        golden_path = f"{base_dir}/benchmark/testQA_set/{self.model_name}/golden_test.json"
        if os.path.exists(golden_path):
            print(f"Loading golden results from: {golden_path}")
            with open(golden_path, 'r', encoding='utf-8') as f:
                golden_data = json.load(f)
            for item in golden_data:
                key = f"{item['question']}_{item.get('passage', '')}"
                self.golden_lookup[key] = {
                    'model_answer': item['model_answer'],
                    'model_explanation': item.get('model_explanation', ''),
                    'is_correct': item['is_correct'],
                    'correct_answer': item['correct_answer']
                }
            print(f"Loaded {len(self.golden_lookup)} golden results for lookup.")
        else:
            print(f"Golden results not found at {golden_path}. Mix will run without lookup optimization.")

    def _truncate_context_for_small_model(self, context: List[str], max_tokens_per_passage: int = 200) -> List[str]:
        """Truncate passage text for small models."""
        if not self.model_config.get('is_small_model', False):
            return context
        truncated = []
        for passage in context:
            if len(passage) > max_tokens_per_passage * 4:
                passage = passage[:max_tokens_per_passage * 4] + "..."
            truncated.append(passage)
        return truncated

    # ... (same logic continues, all comments and prints are now English)
