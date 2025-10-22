#!/usr/bin/env python3
"""
分析MMLU-PRO评估结果并生成模型排行榜
"""
import json
import os
from typing import Dict, List, Tuple

def load_model_results(base_dir: str) -> Dict[str, Dict]:
    """加载所有模型的MMLU-PRO评估结果"""
    results = {}
    
    # 遍历所有模型目录
    local_mcq_dir = os.path.join(base_dir, "DATA", "local_MCQ")
    
    if not os.path.exists(local_mcq_dir):
        print(f"❌ 目录不存在: {local_mcq_dir}")
        return results
    
    for model_name in os.listdir(local_mcq_dir):
        model_dir = os.path.join(local_mcq_dir, model_name)
        if not os.path.isdir(model_dir):
            continue
            
        result_file = os.path.join(model_dir, "mmlu_pro_subjects_test.json")
        if os.path.exists(result_file):
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    model_results = json.load(f)
                
                # 计算准确率
                total_questions = len(model_results)
                correct_answers = sum(1 for r in model_results if r.get('is_correct', False))
                accuracy = (correct_answers / total_questions) * 100 if total_questions > 0 else 0
                
                results[model_name] = {
                    'total_questions': total_questions,
                    'correct_answers': correct_answers,
                    'accuracy': accuracy,
                    'results': model_results
                }
                
                print(f"✅ 加载 {model_name}: {accuracy:.2f}% ({correct_answers}/{total_questions})")
                
            except Exception as e:
                print(f"❌ 加载 {model_name} 失败: {e}")
    
    return results

def generate_leaderboard(results: Dict[str, Dict]) -> List[Dict]:
    """生成模型排行榜"""
    if not results:
        print("❌ 没有找到任何模型结果")
        return []
    
    # 准备数据
    leaderboard_data = []
    for model_name, data in results.items():
        leaderboard_data.append({
            'Model': model_name,
            'Accuracy (%)': round(data['accuracy'], 2),
            'Correct': data['correct_answers'],
            'Total': data['total_questions']
        })
    
    # 按准确率排序
    leaderboard_data.sort(key=lambda x: x['Accuracy (%)'], reverse=True)
    
    # 添加排名
    for i, item in enumerate(leaderboard_data):
        item['Rank'] = i + 1
    
    return leaderboard_data

def analyze_by_subject(results: Dict[str, Dict]) -> Dict[str, Dict]:
    """按学科分析模型性能"""
    subject_analysis = {}
    
    # 获取所有学科
    all_subjects = set()
    for model_name, data in results.items():
        for result in data['results']:
            if 'subject' in result:
                all_subjects.add(result['subject'])
    
    # 按学科分析每个模型
    for subject in all_subjects:
        subject_analysis[subject] = {}
        for model_name, data in results.items():
            subject_results = [r for r in data['results'] if r.get('subject') == subject]
            if subject_results:
                total = len(subject_results)
                correct = sum(1 for r in subject_results if r.get('is_correct', False))
                accuracy = (correct / total) * 100 if total > 0 else 0
                subject_analysis[subject][model_name] = {
                    'accuracy': accuracy,
                    'correct': correct,
                    'total': total
                }
    
    return subject_analysis

def generate_report(leaderboard: List[Dict], subject_analysis: Dict[str, Dict], output_dir: str):
    """生成详细报告"""
    os.makedirs(output_dir, exist_ok=True)
    
    report_file = os.path.join(output_dir, 'mmlu_pro_leaderboard_report.md')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# MMLU-PRO 特定学科模型性能排行榜\n\n")
        f.write(f"**评估时间**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**总模型数**: {len(leaderboard)}\n\n")
        
        if leaderboard:
            f.write(f"**最高准确率**: {leaderboard[0]['Accuracy (%)']:.2f}% ({leaderboard[0]['Model']})\n\n")
            avg_accuracy = sum(item['Accuracy (%)'] for item in leaderboard) / len(leaderboard)
            f.write(f"**平均准确率**: {avg_accuracy:.2f}%\n\n")
        
        f.write("## 总体排行榜\n\n")
        f.write("| 排名 | 模型 | 准确率(%) | 正确题数 | 总题数 |\n")
        f.write("|------|------|-----------|----------|--------|\n")
        for item in leaderboard:
            f.write(f"| {item['Rank']} | {item['Model']} | {item['Accuracy (%)']:.2f} | {item['Correct']} | {item['Total']} |\n")
        f.write("\n")
        
        f.write("## 各学科详细分析\n\n")
        for subject, model_scores in subject_analysis.items():
            f.write(f"### {subject}\n\n")
            f.write("| 模型 | 准确率(%) | 正确题数 | 总题数 |\n")
            f.write("|------|-----------|----------|--------|\n")
            
            # 按准确率排序
            sorted_models = sorted(model_scores.items(), key=lambda x: x[1]['accuracy'], reverse=True)
            for model_name, scores in sorted_models:
                f.write(f"| {model_name} | {scores['accuracy']:.2f} | {scores['correct']} | {scores['total']} |\n")
            f.write("\n")
    
    print(f"✅ 详细报告已保存到: {report_file}")

def save_leaderboard_csv(leaderboard: List[Dict], output_dir: str):
    """保存排行榜为CSV格式"""
    import csv
    
    csv_file = os.path.join(output_dir, 'mmlu_pro_leaderboard.csv')
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        if leaderboard:
            writer = csv.DictWriter(f, fieldnames=leaderboard[0].keys())
            writer.writeheader()
            writer.writerows(leaderboard)
    
    print(f"✅ 排行榜CSV已保存到: {csv_file}")

def main():
    base_dir = "/home/shared/RAG_DATA"
    output_dir = os.path.join(base_dir, "mmlu_pro_analysis_results")
    
    print("正在加载MMLU-PRO评估结果...")
    results = load_model_results(base_dir)
    
    if not results:
        print("❌ 没有找到任何模型结果")
        print("请先运行MMLU-PRO评估脚本")
        return
    
    print(f"\n找到 {len(results)} 个模型的评估结果")
    
    # 生成排行榜
    print("\n生成模型排行榜...")
    leaderboard = generate_leaderboard(results)
    
    if not leaderboard:
        print("❌ 无法生成排行榜")
        return
    
    print("\n" + "="*80)
    print("MMLU-PRO 特定学科模型排行榜")
    print("="*80)
    print(f"{'排名':<4} {'模型':<25} {'准确率(%)':<10} {'正确题数':<8} {'总题数':<6}")
    print("-" * 80)
    for item in leaderboard:
        print(f"{item['Rank']:<4} {item['Model']:<25} {item['Accuracy (%)']:<10.2f} {item['Correct']:<8} {item['Total']:<6}")
    print("="*80)
    
    # 按学科分析
    print("\n按学科分析模型性能...")
    subject_analysis = analyze_by_subject(results)
    
    print("\n各学科最佳模型:")
    for subject, model_scores in subject_analysis.items():
        if model_scores:
            best_model = max(model_scores.items(), key=lambda x: x[1]['accuracy'])
            print(f"  {subject}: {best_model[0]} ({best_model[1]['accuracy']:.2f}%)")
    
    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存排行榜CSV
    save_leaderboard_csv(leaderboard, output_dir)
    
    # 生成详细报告
    print("\n生成详细报告...")
    generate_report(leaderboard, subject_analysis, output_dir)
    
    print(f"\n🎉 分析完成！所有结果已保存到: {output_dir}")

if __name__ == "__main__":
    main()