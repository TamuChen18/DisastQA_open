#!/usr/bin/env python3
"""
API模型MMLU-PRO评估脚本
支持GPT-4o和Gemini 1.5 Pro
"""
import json
import os
import argparse
import time
import re
import requests
from typing import Dict, List, Optional
from dotenv import load_dotenv

# API配置
API_CONFIGS = {
    "gpt-4o": {
        "base_url": "https://api.openai.com/v1/chat/completions",
        "headers": {
            "Content-Type": "application/json",
            "Authorization": "Bearer YOUR_OPENAI_API_KEY"  # 需要替换
        },
        "model_name": "gpt-4o"
    },
    "gemini-1.5-pro": {
        "base_url": "https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent",
        "headers": {
            "Content-Type": "application/json"
        },
        "model_name": "gemini-pro",
        "url": ""
    }
}

def load_api_keys():
    """从环境变量加载API密钥"""
    # 加载.env文件
    env_path = "/home/shared/RAG_DATA/DATA/.env"
    if os.path.exists(env_path):
        load_dotenv(env_path)
    
    openai_key = os.getenv('OPENAI_API_KEY')
    google_key = os.getenv('GOOGLE_API_KEY')
    
    if openai_key:
        API_CONFIGS["gpt-4o"]["headers"]["Authorization"] = f"Bearer {openai_key}"
    
    if google_key:
        API_CONFIGS["gemini-1.5-pro"]["url"] = f"{API_CONFIGS['gemini-1.5-pro']['base_url']}?key={google_key}"
    
    return openai_key, google_key

def load_test_data(test_set_path):
    """加载测试数据"""
    with open(test_set_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # 目标学科列表
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
            
            if subject in target_subjects:
                processed_item = {
                    'question': content['question'],
                    'options': content['options'],
                    'correct_answer': content['correct_option'],
                    'subject': subject
                }
                test_set.append(processed_item)
                
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
        answer_match = re.search(r'Answer:\s*([A-J])', response, re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).upper()
        
        # 尝试匹配单独的大写字母
        letter_match = re.search(r'\b([A-J])\b', response)
        if letter_match:
            return letter_match.group(1).upper()
        
        # 尝试匹配选项格式 "A. ..."
        option_match = re.search(r'^([A-J])\.', response.strip(), re.MULTILINE)
        if option_match:
            return option_match.group(1).upper()
        
        return None
    except Exception as e:
        print(f"解析答案时出错: {e}")
        return None

def call_openai_api(prompt, model_name="gpt-4o", max_tokens=5):
    """调用OpenAI API"""
    config = API_CONFIGS[model_name]
    
    payload = {
        "model": config["model_name"],
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0
    }
    
    try:
        response = requests.post(
            config["base_url"],
            headers=config["headers"],
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"OpenAI API调用失败: {e}")
        return ""

def call_gemini_api(prompt, model_name="gemini-1.5-pro", max_tokens=5):
    """调用Gemini API"""
    config = API_CONFIGS[model_name]
    
    payload = {
        "contents": [{
            "role": "user",
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": max_tokens
        }
    }
    
    try:
        response = requests.post(
            config["url"],
            headers=config["headers"],
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        if "candidates" in result and result["candidates"]:
            candidate = result["candidates"][0]
            finish_reason = candidate.get("finishReason", "")
            
            # Handle MAX_TOKENS case
            if finish_reason == "MAX_TOKENS":
                return "Unable to complete response due to token limit"
            
            if "content" in candidate:
                content = candidate["content"]
                if "parts" in content and content["parts"]:
                    return content["parts"][0]["text"].strip()
                elif "text" in content:
                    return content["text"].strip()
                elif content.get("role") == "model" and not content.get("parts"):
                    return "Unable to generate response - empty model output"
            elif "text" in candidate:
                return candidate["text"].strip()
        
        return "Unexpected response format"
    except Exception as e:
        print(f"Gemini API调用失败: {e}")
        return ""

def generate_answer(model_name, prompt, max_new_tokens=5):
    """使用API生成答案"""
    if model_name == "gpt-4o":
        return call_openai_api(prompt, model_name, max_new_tokens)
    elif model_name == "gemini-1.5-pro":
        return call_gemini_api(prompt, model_name, max_new_tokens)
    else:
        print(f"不支持的模型: {model_name}")
        return ""

def evaluate_model(model_name, test_set, max_new_tokens=5, test_mode=False):
    """评估单个模型"""
    print(f"\n{'='*60}")
    print(f"评估模型: {model_name}")
    print(f"{'='*60}")
    
    # 检查API密钥
    openai_key, google_key = load_api_keys()
    
    if model_name == "gpt-4o" and not openai_key:
        print("❌ 未找到OpenAI API密钥，请设置OPENAI_API_KEY环境变量")
        return False
    
    if model_name == "gemini-1.5-pro" and not google_key:
        print("❌ 未找到Google API密钥，请设置GOOGLE_API_KEY环境变量")
        return False
    
    # 限制测试数据
    if test_mode:
        test_set = test_set[:10]
        print(f"测试模式：只处理前 {len(test_set)} 个问题")
    
    results = []
    correct_count = 0
    subject_stats = {}
    
    total_questions = len(test_set)
    
    for i, item in enumerate(test_set, 1):
        question = item['question']
        options = item['options']
        correct_answer = item['correct_answer']
        subject = item['subject']
        
        print(f"\n处理问题 {i}/{total_questions}")
        print(f"学科: {subject}")
        print(f"问题: {question[:100]}...")
        
        # 创建prompt
        prompt = create_prompt(question, options)
        print(f"Prompt长度: {len(prompt)}")
        
        # 生成答案
        print("正在生成答案...")
        start_time = time.time()
        
        try:
            model_response = generate_answer(model_name, prompt, max_new_tokens)
            generation_time = time.time() - start_time
            
            print(f"模型回答: {model_response[:100]}...")
            
            # 解析答案
            parsed_answer = parse_answer(model_response)
            
            if parsed_answer:
                print(f"正确答案: {correct_answer}")
                print(f"模型答案: {parsed_answer}")
                is_correct = parsed_answer == correct_answer
                print(f"正确: {is_correct}")
                
                if is_correct:
                    correct_count += 1
            else:
                print("Error parsing answer: No valid answer found in response")
                parsed_answer = ""
                is_correct = False
            
            # 统计各学科
            if subject not in subject_stats:
                subject_stats[subject] = {'correct': 0, 'total': 0}
            subject_stats[subject]['total'] += 1
            if is_correct:
                subject_stats[subject]['correct'] += 1
            
            # 保存结果
            result_item = {
                'question': question,
                'options': options,
                'correct_answer': correct_answer,
                'model_answer': parsed_answer,
                'model_response': model_response,
                'is_correct': is_correct,
                'subject': subject,
                'generation_time': generation_time
            }
            results.append(result_item)
            
        except Exception as e:
            print(f"处理问题时出错: {e}")
            continue
    
    # 计算准确率
    accuracy = (correct_count / total_questions) * 100 if total_questions > 0 else 0
    
    # 保存结果
    output_dir = f"/home/shared/RAG_DATA/DATA/local_MCQ/{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "mmlu_pro_subjects_test.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到: {output_file}")
    
    # 打印统计结果
    print(f"\n统计结果:")
    print(f"总问题数: {total_questions}")
    print(f"正确答案数: {correct_count}")
    print(f"总体准确率: {accuracy:.2f}%")
    
    print(f"\n各学科准确率:")
    for subject, stats in subject_stats.items():
        subject_accuracy = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
        print(f"  {subject}: {subject_accuracy:.2f}% ({stats['correct']}/{stats['total']})")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='API模型MMLU-PRO评估')
    parser.add_argument('--model', type=str, help='模型名称 (gpt-4o, gemini-1.5-pro)')
    parser.add_argument('--test_mode', action='store_true', help='测试模式，只处理前10个问题')
    parser.add_argument('--max_new_tokens', type=int, default=5, help='最大生成token数')
    
    args = parser.parse_args()
    
    # 加载测试数据
    test_set_path = "/home/shared/RAG_DATA/DATA/MMLUE-PRO/mmlu_pro_subjects_mcq_format.json"
    if not os.path.exists(test_set_path):
        print(f"❌ 测试数据文件不存在: {test_set_path}")
        return
    
    print("正在加载测试数据...")
    test_set = load_test_data(test_set_path)
    
    if not test_set:
        print("❌ 没有找到测试数据")
        return
    
    # 评估指定模型
    if args.model:
        if args.model not in API_CONFIGS:
            print(f"❌ 不支持的模型: {args.model}")
            print(f"支持的模型: {list(API_CONFIGS.keys())}")
            return
        
        evaluate_model(args.model, test_set, args.max_new_tokens, args.test_mode)
    else:
        print("请指定要评估的模型: --model gpt-4o 或 --model gemini-1.5-pro")

if __name__ == "__main__":
    main()
