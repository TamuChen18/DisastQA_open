#!/usr/bin/env python3
"""
生成按难度分类的Coverage表格数据
"""
import json
import pandas as pd

def load_difficulty_data(file_path):
    """加载按难度分类的数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['results']

def generate_difficulty_table(data):
    """生成按难度分类的表格数据"""
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    # 模型名称映射（保持与表格一致）
    model_name_mapping = {
        'gpt-4o': 'GPT-4o',
        'gemini-1.5-pro': 'Gemini-1.5 Pro',
        'qwen-3-8b': 'Qwen-3-8B',
        'Yi-6B-Chat': 'Yi-6B-Chat',
        'llama-3-8b': 'Llama-3-8B',
        'qwen-3-4b': 'Qwen-3-4B',
        'qwen-2.5-3b-instruct': 'Qwen-2.5-3B-Instr.',
        'phi-2': 'Phi-2',
        'Mistral-7B-Instruct-v0.2': 'Mistral-7B-Instr.',
        'gemma-7b': 'Gemma-7B',
        'deepseek-v3-7b': 'DeepSeek-v3-7B',
        'llama-3.2-3b-instruct': 'Llama-3.2-3B-Instr.',
        'Llama-3.2-1B-Instruct': 'Llama-3.2-1B-Instr.',
        'Falcon3-1B-Instruct': 'Falcon-3-1B-Instr.',
        'AceMath-1.5B-Instruct': 'AceMath-1.5B-Instr.',
        'Hunyuan-4B-Instruct': 'Hunyuan-4B-Instr.',
        'Hunyuan-7B-Instruct': 'Hunyuan-7B-Instr.',
        'qwen-3-0.6b': 'Qwen-3-0.6B'
    }
    
    # 难度级别映射
    difficulty_mapping = {
        'easy': 'Easy',
        'medium': 'Medium', 
        'hard': 'Hard',
        'extremely_complex': 'Extremely Complex'
    }
    
    # 测试类型映射
    test_type_mapping = {
        'base': 'Base',
        'mix': 'Mix',
        'golden': 'Golden'
    }
    
    # 应用映射
    df['model_display'] = df['model'].map(model_name_mapping)
    df['difficulty_display'] = df['difficulty'].map(difficulty_mapping)
    df['test_type_display'] = df['test_type'].map(test_type_mapping)
    
    # 创建透视表
    pivot_table = df.pivot_table(
        index='model_display',
        columns=['difficulty_display', 'test_type_display'],
        values='avg_coverage',
        aggfunc='mean'
    )
    
    # 重新排列列的顺序
    difficulty_order = ['Easy', 'Medium', 'Hard', 'Extremely Complex']
    test_type_order = ['Base', 'Mix', 'Golden']
    
    # 重新排列列
    new_columns = []
    for difficulty in difficulty_order:
        for test_type in test_type_order:
            if (difficulty, test_type) in pivot_table.columns:
                new_columns.append((difficulty, test_type))
    
    pivot_table = pivot_table[new_columns]
    
    # 按模型名称排序（与表格顺序一致）
    model_order = [
        'GPT-4o', 'Gemini-1.5 Pro', 'Qwen-3-8B', 'Yi-6B-Chat', 'Llama-3-8B',
        'Qwen-3-4B', 'Qwen-2.5-3B-Instr.', 'Phi-2', 'Mistral-7B-Instr.',
        'Gemma-7B', 'DeepSeek-v3-7B', 'Llama-3.2-3B-Instr.', 'Llama-3.2-1B-Instr.',
        'Falcon-3-1B-Instr.', 'AceMath-1.5B-Instr.', 'Hunyuan-4B-Instr.',
        'Hunyuan-7B-Instr.', 'Qwen-3-0.6B'
    ]
    
    # 只保留存在的模型
    existing_models = [model for model in model_order if model in pivot_table.index]
    pivot_table = pivot_table.loc[existing_models]
    
    return pivot_table

def format_table_for_latex(pivot_table):
    """格式化为LaTeX表格格式"""
    
    print("\\begin{table*}[t]")
    print("\\centering")
    print("\\small")
    print("\\setlength{\\tabcolsep}{3pt}")
    print("\\renewcommand{\\arraystretch}{1.05}")
    print("\\resizebox{\\textwidth}{!}{%")
    print("\\begin{tabular}{l|ccc|ccc|ccc|ccc}")
    print("\\toprule")
    print("\\multirow{2}{*}{\\textbf{Model}} & ")
    print("\\multicolumn{3}{c|}{\\textbf{Easy}} & ")
    print("\\multicolumn{3}{c|}{\\textbf{Medium}} & ")
    print("\\multicolumn{3}{c|}{\\textbf{Hard}} & ")
    print("\\multicolumn{3}{c}{\\textbf{Extremely Complex}} \\\\")
    print("\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10} \\cmidrule(lr){11-13}")
    print(" & Base & Mix & Golden & Base & Mix & Golden & Base & Mix & Golden & Base & Mix & Golden \\\\")
    print("\\midrule")
    
    for model in pivot_table.index:
        row = f"{model:<25}"
        for difficulty in ['Easy', 'Medium', 'Hard', 'Extremely Complex']:
            for test_type in ['Base', 'Mix', 'Golden']:
                if (difficulty, test_type) in pivot_table.columns:
                    value = pivot_table.loc[model, (difficulty, test_type)]
                    if pd.notna(value):
                        row += f" & {value:.2f}"
                    else:
                        row += " & N/A"
                else:
                    row += " & N/A"
        row += " \\\\"
        print(row)
    
    print("\\bottomrule")
    print("\\end{tabular}}")
    print("\\caption{")
    print("Full breakdown of \\textbf{performance measured by keypoint coverage (\\%)} across difficulty levels (Easy, Medium, Hard, Extremely Complex) under \\textbf{Base}, \\textbf{Mix}, and \\textbf{Golden} retrieval settings for all 18 models.")
    print("}")
    print("\\label{tab:difficulty_full}")
    print("\\end{table*}")

def main():
    # 加载数据
    data = load_difficulty_data('/home/shared/RAG_DATA/oe_difficulty_performance_data.json')
    
    # 生成表格
    pivot_table = generate_difficulty_table(data)
    
    # 显示表格
    print("按难度分类的Coverage表格数据:")
    print("=" * 80)
    print(pivot_table.round(2))
    
    print("\n" + "=" * 80)
    print("LaTeX格式:")
    print("=" * 80)
    
    # 格式化为LaTeX
    format_table_for_latex(pivot_table)
    
    # 保存为CSV
    pivot_table.to_csv('/home/shared/RAG_DATA/difficulty_coverage_table.csv')
    print(f"\n表格已保存到: /home/shared/RAG_DATA/difficulty_coverage_table.csv")

if __name__ == "__main__":
    main()
