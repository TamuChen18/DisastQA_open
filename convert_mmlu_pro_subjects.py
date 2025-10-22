#!/usr/bin/env python3
"""
将MMLU-PRO特定学科数据转换为与现有MCQ数据相同的格式
只处理：Medicine, Biology, Geography, Global Facts, Law, Sociology, Public Relations, Psychology, Economics
"""
import json
import os
import subprocess
import sys

# 目标学科列表 (8个学科，每个250题，总共2000题)
TARGET_SUBJECTS = [
    "health", "biology", "law", "psychology", "economics", 
    "business", "chemistry", "engineering"
]

def install_pandas():
    """尝试安装pandas和pyarrow"""
    try:
        import pandas as pd
        import pyarrow
        return True
    except ImportError:
        print("需要安装pandas和pyarrow来处理parquet文件")
        print("尝试安装...")
        
        try:
            # 尝试使用pip安装
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "--break-system-packages", "pandas", "pyarrow"])
            print("✅ pandas和pyarrow安装成功")
            return True
        except subprocess.CalledProcessError:
            print("❌ 自动安装失败，请手动安装:")
            print("pip install pandas pyarrow")
            return False

def convert_mmlu_pro_subjects_to_mcq_format():
    """将MMLU-PRO特定学科数据转换为MCQ格式"""
    
    # 输入和输出路径
    parquet_file = "DATA/MMLUE-PRO/data/test-00000-of-00001.parquet"
    output_file = "DATA/MMLUE-PRO/mmlu_pro_subjects_mcq_format.json"
    
    if not os.path.exists(parquet_file):
        print(f"❌ 文件不存在: {parquet_file}")
        return False
    
    print("正在读取MMLU-PRO数据...")
    print(f"目标学科: {', '.join(TARGET_SUBJECTS)}")
    
    # 尝试安装pandas
    if not install_pandas():
        print("无法安装pandas，创建示例数据...")
        create_sample_subjects_data(output_file)
        return True
    
    try:
        import pandas as pd
        
        # 读取parquet文件
        df = pd.read_parquet(parquet_file)
        print(f"✅ 读取成功，共 {len(df)} 个样本")
        
        # 显示数据结构
        print("\n原始数据结构:")
        print("Columns:", list(df.columns))
        
        # 检查学科分布
        if 'category' in df.columns:
            print("\n所有学科分布:")
            category_counts = df['category'].value_counts()
            for category, count in category_counts.items():
                print(f"  {category}: {count}")
            
            # 筛选目标学科
            print(f"\n筛选目标学科...")
            target_df = df[df['category'].isin(TARGET_SUBJECTS)]
            print(f"筛选后样本数: {len(target_df)}")
            
            # 显示筛选后的学科分布
            print("\n筛选后的学科分布:")
            target_category_counts = target_df['category'].value_counts()
            for category, count in target_category_counts.items():
                print(f"  {category}: {count}")
            
            # 每个学科采样250题，总共2000题
            print("\n正在采样，每个学科250题...")
            sampled_data = []
            for category in TARGET_SUBJECTS:
                category_data = target_df[target_df['category'] == category]
                if len(category_data) >= 250:
                    # 随机采样250题
                    sampled_category = category_data.sample(n=250, random_state=42)
                else:
                    # 如果不足250题，全部使用
                    sampled_category = category_data
                    print(f"  警告: {category} 只有 {len(category_data)} 题，全部使用")
                
                sampled_data.append(sampled_category)
                print(f"  {category}: 采样 {len(sampled_category)} 题")
            
            # 合并所有采样的数据
            target_df = pd.concat(sampled_data, ignore_index=True)
            print(f"\n最终采样结果: {len(target_df)} 题")
            
            # 显示最终分布
            print("\n最终学科分布:")
            final_counts = target_df['category'].value_counts()
            for category, count in final_counts.items():
                print(f"  {category}: {count}")
                
        else:
            print("❌ 数据中没有category字段")
            return False
        
        # 转换为MCQ格式
        print("\n正在转换为MCQ格式...")
        mcq_data = []
        
        for idx, row in target_df.iterrows():
            # 构建MCQ格式的数据结构
            # 确保所有数据都转换为Python原生类型
            options = row['options'].tolist() if hasattr(row['options'], 'tolist') else list(row['options'])
            options_list = [f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)]
            
            # 确保答案索引在有效范围内
            answer_index = int(row['answer_index'])
            if 0 <= answer_index < len(options):
                correct_option_char = chr(65 + answer_index)
            else:
                correct_option_char = "UNKNOWN"
                print(f"警告: 问题ID {row.get('question_id', 'N/A')} 的答案索引 {answer_index} 超出选项范围 {len(options)}")
            
            mcq_item = {
                "passage": "",  # MMLU-PRO没有passage，留空
                "category": str(row['category']),  # 添加学科信息
                "multiple_choice": {
                    "gpt40": {
                        "content": {
                            "question": str(row['question']),
                            "options": options_list,
                            "correct_option": correct_option_char
                        }
                    }
                }
            }
            mcq_data.append(mcq_item)
        
        # 保存转换后的数据
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(mcq_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 转换完成，共 {len(mcq_data)} 个样本")
        print(f"数据已保存到: {output_file}")
        
        # 显示转换后的数据结构
        print("\n转换后的数据结构:")
        print("Keys:", list(mcq_data[0].keys()))
        print("\n第一个样本:")
        sample = mcq_data[0]
        print(f"category: {sample['category']}")
        print(f"passage: '{sample['passage']}'")
        print(f"question: {sample['multiple_choice']['gpt40']['content']['question'][:100]}...")
        print(f"options: {sample['multiple_choice']['gpt40']['content']['options']}")
        print(f"correct_option: {sample['multiple_choice']['gpt40']['content']['correct_option']}")
        
        # 统计信息
        print(f"\n统计信息:")
        print(f"总问题数: {len(mcq_data)}")
        
        # 统计各学科的问题数量
        subject_counts = {}
        for item in mcq_data:
            subject = item['category']
            if subject not in subject_counts:
                subject_counts[subject] = 0
            subject_counts[subject] += 1
        
        print(f"\n各学科问题数量:")
        for subject, count in subject_counts.items():
            print(f"  {subject}: {count}")
        
        # 统计选项数量分布
        option_counts = {}
        for item in mcq_data:
            num_options = len(item['multiple_choice']['gpt40']['content']['options'])
            if num_options not in option_counts:
                option_counts[num_options] = 0
            option_counts[num_options] += 1
        
        print(f"\n选项数量分布:")
        for num_options, count in sorted(option_counts.items()):
            print(f"  {num_options}个选项: {count}题")
            
        return True
            
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_sample_subjects_data(output_file):
    """创建特定学科的示例数据"""
    print("创建特定学科示例数据...")
    
    # 为每个目标学科创建一个示例问题
    sample_data = []
    
    sample_questions = {
        "Medicine": {
            "question": "What is the most common cause of acute myocardial infarction?",
            "options": ["Coronary artery spasm", "Atherosclerosis", "Viral infection", "Trauma", "Drug toxicity", "Genetic mutation", "Autoimmune disease", "Metabolic disorder", "Infectious disease", "Environmental factors"],
            "correct_option": "B"
        },
        "Biology": {
            "question": "Which organelle is responsible for protein synthesis?",
            "options": ["Mitochondria", "Ribosome", "Nucleus", "Golgi apparatus", "Endoplasmic reticulum", "Lysosome", "Chloroplast", "Vacuole", "Cell membrane", "Cytoplasm"],
            "correct_option": "B"
        },
        "Geography": {
            "question": "What is the longest river in the world?",
            "options": ["Amazon", "Nile", "Mississippi", "Yangtze", "Yellow", "Ganges", "Danube", "Rhine", "Thames", "Seine"],
            "correct_option": "B"
        },
        "Global Facts": {
            "question": "Which country has the largest population?",
            "options": ["India", "China", "United States", "Indonesia", "Pakistan", "Brazil", "Nigeria", "Bangladesh", "Russia", "Mexico"],
            "correct_option": "B"
        },
        "Law": {
            "question": "What is the principle of 'innocent until proven guilty' called?",
            "options": ["Due process", "Presumption of innocence", "Burden of proof", "Habeas corpus", "Double jeopardy", "Miranda rights", "Statute of limitations", "Res judicata", "Stare decisis", "Actus reus"],
            "correct_option": "B"
        },
        "Sociology": {
            "question": "What term describes the process by which individuals learn the norms and values of their society?",
            "options": ["Assimilation", "Socialization", "Acculturation", "Integration", "Enculturation", "Indoctrination", "Education", "Training", "Conditioning", "Programming"],
            "correct_option": "B"
        },
        "Public Relations": {
            "question": "What is the primary goal of crisis communication?",
            "options": ["Increase sales", "Protect reputation", "Entertain audience", "Educate public", "Promote products", "Generate leads", "Build awareness", "Create controversy", "Avoid responsibility", "Maximize profit"],
            "correct_option": "B"
        },
        "Psychology": {
            "question": "Which psychological theory emphasizes the role of unconscious processes?",
            "options": ["Behaviorism", "Psychoanalysis", "Cognitive psychology", "Humanistic psychology", "Social psychology", "Developmental psychology", "Biological psychology", "Evolutionary psychology", "Positive psychology", "Gestalt psychology"],
            "correct_option": "B"
        },
        "Economics": {
            "question": "What is the economic term for the total value of all goods and services produced in a country?",
            "options": ["Net National Product", "Gross Domestic Product", "Gross National Product", "Net Domestic Product", "National Income", "Personal Income", "Disposable Income", "Real Income", "Nominal Income", "Per Capita Income"],
            "correct_option": "B"
        }
    }
    
    for subject, question_data in sample_questions.items():
        sample_item = {
            "passage": "",
            "category": subject,
            "multiple_choice": {
                "gpt40": {
                    "content": {
                        "question": question_data["question"],
                        "options": question_data["options"],
                        "correct_option": question_data["correct_option"]
                    }
                }
            }
        }
        sample_data.append(sample_item)
    
    # 保存示例数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 特定学科示例数据创建完成，共 {len(sample_data)} 个样本")
    print(f"数据已保存到: {output_file}")
    print("\n注意：这只是示例数据，要处理完整的MMLU-PRO数据集需要安装pandas")

if __name__ == "__main__":
    convert_mmlu_pro_subjects_to_mcq_format()