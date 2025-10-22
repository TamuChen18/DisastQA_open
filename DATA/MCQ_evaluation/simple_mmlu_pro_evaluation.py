#!/usr/bin/env python3
"""
简化版MMLU-PRO特定学科评估脚本
不依赖额外的包，只使用标准库
"""
import json
import os
import argparse
import time
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_configs():
    """加载模型配置"""
    return {
        "llama-3-8b": {
            "path": "/home/shared/RAG_DATA/DATA/models/llama-3-8b",
            "max_sequence_length": 8192,
            "max_new_tokens": 32,
        },
        "llama-3.2-3b-instruct": {
            "path": "/home/shared/RAG_DATA/DATA/models/llama-3.2-3b-instruct",
            "max_sequence_length": 8192,
            "max_new_tokens": 32,
        },
        "Mistral-7B-Instruct-v0.2": {
            "path": "/home/shared/RAG_DATA/DATA/models/Mistral-7B-Instruct-v0.2",
            "max_sequence_length": 8192,
            "max_new_tokens": 32,
        },
        "qwen-2.5-3b-instruct": {
            "path": "/home/shared/RAG_DATA/DATA/models/qwen-2.5-3b-instruct",
            "max_sequence_length": 8192,
            "max_new_tokens": 32,
        },
        "qwen-3-4b": {
            "path": "/home/shared/RAG_DATA/DATA/models/qwen-3-4b",
            "max_sequence_length": 8192,
            "max_new_tokens": 32,
        },
        "qwen-3-8b": {
            "path": "/home/shared/RAG_DATA/DATA/models/qwen-3-8b",
            "max_sequence_length": 8192,
            "max_new_tokens": 32,
        },
        "deepseek-v3-7b": {
            "path": "/home/shared/RAG_DATA/DATA/models/deepseek-v3-7b",
            "max_sequence_length": 8192,
            "max_new_tokens": 32,
        },
        "phi-2": {
            "path": "/home/shared/RAG_DATA/DATA/models/phi-2",
            "max_sequence_length": 2048,
            "max_new_tokens": 32,
        },
        "gemma-7b": {
            "path": "/home/shared/RAG_DATA/DATA/models/gemma-7b",
            "max_sequence_length": 2048,
            "max_new_tokens": 32,
        },
        "Llama-3.2-1B-Instruct": {
            "path": "/home/shared/RAG_DATA/DATA/models/Llama-3.2-1B-Instruct",
            "max_sequence_length": 2048,
            "max_new_tokens": 32,
        },
        "qwen-3-0.6b": {
            "path": "/home/shared/RAG_DATA/DATA/models/qwen-3-0.6b",
            "max_sequence_length": 8192,
            "max_new_tokens": 32,
        },
        "Hunyuan-7B-Instruct": {
            "path": "/home/shared/RAG_DATA/DATA/models/Hunyuan-7B-Instruct",
            "max_sequence_length": 8192,
            "max_new_tokens": 32,
        },
        "Hunyuan-4B-Instruct": {
            "path": "/home/shared/RAG_DATA/DATA/models/Hunyuan-4B-Instruct",
            "max_sequence_length": 8192,
            "max_new_tokens": 32,
        },
        "Yi-6B-Chat": {
            "path": "/home/shared/RAG_DATA/DATA/models/Yi-6B-Chat",
            "max_sequence_length": 8192,
            "max_new_tokens": 32,
        },
        "AceMath-1.5B-Instruct": {
            "path": "/home/shared/RAG_DATA/DATA/models/AceMath-1.5B-Instruct",
            "max_sequence_length": 2048,
            "max_new_tokens": 32,
        },
        "Falcon3-1B-Instruct": {
            "path": "/home/shared/RAG_DATA/DATA/models/Falcon3-1B-Instruct",
            "max_sequence_length": 2048,
            "max_new_tokens": 32,
        }
    }

def load_test_data(test_set_path):
    """加载测试数据"""
    with open(test_set_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # 目标学科列表 (使用实际数据中的学科名称)
    target_subjects = [
        "health", "biology", "law", "psychology", "economics", 
        "business", "chemistry", "engineering"
    ]
    
    test_set = []
    subject_stats = {}
    
    for item in raw_data:
        if 'multiple_choice' in item and 'gpt40' in item['multiple_choice']:
            content = item['multiple_choice']['gpt40']['content']
            subject = item.get('category', 'Unknown')
            
            # 只处理目标学科
            if subject in target_subjects:
                processed_item = {
                    'question': content['question'],
                    'options': content['options'],
                    'correct_answer': content['correct_option'],
                    'subject': subject
                }
                test_set.append(processed_item)
                
                # 统计各学科数量
                if subject not in subject_stats:
                    subject_stats[subject] = 0
                subject_stats[subject] += 1
    
    print(f"\n筛选后的数据统计:")
    print(f"总问题数: {len(test_set)}")
    for subject, count in subject_stats.items():
        print(f"  {subject}: {count}题")
    
    return test_set

def create_prompt(question, options):
    """创建与现有MCQ完全一致的prompt"""
    options_str = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
    prompt = f"""Question: {question}\nOptions:\n{options_str}\n\nAnswer:"""
    return prompt

def parse_answer(response):
    """解析模型答案"""
    try:
        # 首先尝试匹配 "Answer: X" 格式
        answer_match = re.search(r'Answer:\s*([A-J])', response)
        if answer_match:
            answer = answer_match.group(1)
        else:
            # 如果没有找到标准格式，尝试直接匹配 A-J
            answer_match = re.search(r'[A-J]', response)
            if answer_match:
                answer = answer_match.group(0)
            else:
                # 如果还是没有找到，尝试匹配选项前缀
                answer_match = re.search(r'([A-J])\.', response)
                if answer_match:
                    answer = answer_match.group(1)
                else:
                    raise ValueError("No valid answer found in response")
        
        return answer
    except Exception as e:
        print(f"Error parsing answer: {str(e)}")
        return ''

def load_model(model_name, model_configs):
    """加载模型和tokenizer"""
    config = model_configs[model_name]
    model_path = config['path']
    
    print(f"正在加载模型: {model_name}")
    print(f"模型路径: {model_path}")
    
    try:
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
        
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print("✅ 模型加载成功!")
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None, None

def generate_answer(model, tokenizer, prompt, max_new_tokens=5):
    """使用模型生成答案"""
    try:
        # 编码输入
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            return_token_type_ids=False  # 明确禁用token_type_ids
        ).to(model.device)
        
        # 生成回答
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # 解码输出
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 移除prompt部分
        response = response[len(prompt):].strip()
        
        return response
        
    except Exception as e:
        print(f"生成答案时出错: {e}")
        return ""

def real_model_evaluation(test_set, model_name, model_configs):
    """真实模型评估"""
    print(f"\n{'='*60}")
    print(f"评估模型: {model_name}")
    print(f"{'='*60}")
    
    # 加载模型
    model, tokenizer = load_model(model_name, model_configs)
    if model is None or tokenizer is None:
        print(f"❌ 无法加载模型 {model_name}，跳过")
        return []
    
    results = []
    total = len(test_set)
    
    for i, item in enumerate(test_set):
        print(f"\n处理问题 {i+1}/{total}")
        print(f"学科: {item['subject']}")
        print(f"问题: {item['question'][:100]}...")
        
        # 创建prompt
        prompt = create_prompt(item['question'], item['options'])
        print(f"Prompt长度: {len(prompt)}")
        
        # 生成答案
        print("正在生成答案...")
        response = generate_answer(model, tokenizer, prompt)
        print(f"模型回答: {response[:200]}...")
        
        # 解析答案
        parsed_answer = parse_answer(response)
        
        # 验证答案
        is_correct = parsed_answer == item['correct_answer']
        
        # 保存结果
        result = {
            'question': item['question'],
            'options': item['options'],
            'correct_answer': item['correct_answer'],
            'model_answer': parsed_answer,
            'model_response': response,
            'is_correct': is_correct,
            'subject': item['subject']
        }
        results.append(result)
        
        print(f"正确答案: {item['correct_answer']}")
        print(f"模型答案: {parsed_answer}")
        print(f"正确: {is_correct}")
        
        # 清理GPU内存
        torch.cuda.empty_cache()
    
    return results

def save_results(results, output_path):
    """保存结果"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {output_path}")

def calculate_statistics(results):
    """计算统计信息"""
    total_questions = len(results)
    correct_answers = sum(1 for r in results if r['is_correct'])
    accuracy = (correct_answers / total_questions) * 100 if total_questions > 0 else 0
    
    print(f"\n统计结果:")
    print(f"总问题数: {total_questions}")
    print(f"正确答案数: {correct_answers}")
    print(f"总体准确率: {accuracy:.2f}%")
    
    # 按学科统计
    subject_stats = {}
    for result in results:
        subject = result['subject']
        if subject not in subject_stats:
            subject_stats[subject] = {'total': 0, 'correct': 0}
        subject_stats[subject]['total'] += 1
        if result['is_correct']:
            subject_stats[subject]['correct'] += 1
    
    print(f"\n各学科准确率:")
    for subject, stats in subject_stats.items():
        subject_accuracy = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
        print(f"  {subject}: {subject_accuracy:.2f}% ({stats['correct']}/{stats['total']})")

def main():
    parser = argparse.ArgumentParser(description='简化版MMLU-PRO特定学科评估')
    parser.add_argument('--test_mode', action='store_true', help='测试模式（前10个问题）')
    parser.add_argument('--model', type=str, help='指定模型名称')
    args = parser.parse_args()

    base_dir = "/home/shared/RAG_DATA"
    test_set_path = f"{base_dir}/DATA/MMLUE-PRO/mmlu_pro_subjects_mcq_format.json"
    
    # 检查测试集是否存在
    if not os.path.exists(test_set_path):
        print(f"❌ 测试集文件不存在: {test_set_path}")
        print("请先运行数据转换脚本")
        return

    # 加载测试数据
    print("正在加载测试数据...")
    test_set = load_test_data(test_set_path)
    
    if not test_set:
        print("❌ 没有找到测试数据")
        return
    
    # 测试模式
    if args.test_mode:
        test_set = test_set[:10]
        print(f"测试模式：只处理前 {len(test_set)} 个问题")
    
    # 选择模型
    model_configs = load_model_configs()
    if args.model:
        if args.model in model_configs:
            model_names = [args.model]
        else:
            print(f"❌ 模型 {args.model} 不在配置中")
            print(f"可用模型: {list(model_configs.keys())}")
            return
    else:
        model_names = list(model_configs.keys())
    
    # 评估每个模型
    for model_name in model_names:
        print(f"\n开始评估模型: {model_name}")
        
        # 设置输出路径
        output_path = f"{base_dir}/DATA/local_MCQ/{model_name}/mmlu_pro_subjects_test.json"
        if os.path.exists(output_path):
            print(f"输出文件已存在: {output_path}")
            print("跳过此模型...")
            continue
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 真实模型评估
        results = real_model_evaluation(test_set, model_name, model_configs)
        
        # 保存结果
        save_results(results, output_path)
        
        # 计算统计信息
        calculate_statistics(results)

if __name__ == "__main__":
    main()
