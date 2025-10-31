#!/usr/bin/env python3
"""
Analyze MMLU-PRO evaluation results and generate model leaderboard.
"""
import json
import os
from typing import Dict, List, Tuple

def load_model_results(base_dir: str) -> Dict[str, Dict]:
    """Load MMLU-PRO evaluation results for all models."""
    results = {}
    local_mcq_dir = os.path.join(base_dir, "DATA", "local_MCQ")
    
    if not os.path.exists(local_mcq_dir):
        print(f"Directory not found: {local_mcq_dir}")
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
                
                total_questions = len(model_results)
                correct_answers = sum(1 for r in model_results if r.get('is_correct', False))
                accuracy = (correct_answers / total_questions) * 100 if total_questions > 0 else 0
                
                results[model_name] = {
                    'total_questions': total_questions,
                    'correct_answers': correct_answers,
                    'accuracy': accuracy,
                    'results': model_results
                }
                
                print(f"Loaded {model_name}: {accuracy:.2f}% ({correct_answers}/{total_questions})")
                
            except Exception as e:
                print(f"Failed to load {model_name}: {e}")
    
    return results

def generate_leaderboard(results: Dict[str, Dict]) -> List[Dict]:
    """Generate a leaderboard from model results."""
    if not results:
        print("No model results found.")
        return []
    
    leaderboard_data = []
    for model_name, data in results.items():
        leaderboard_data.append({
            'Model': model_name,
            'Accuracy (%)': round(data['accuracy'], 2),
            'Correct': data['correct_answers'],
            'Total': data['total_questions']
        })
    
    leaderboard_data.sort(key=lambda x: x['Accuracy (%)'], reverse=True)
    
    for i, item in enumerate(leaderboard_data):
        item['Rank'] = i + 1
    
    return leaderboard_data

def analyze_by_subject(results: Dict[str, Dict]) -> Dict[str, Dict]:
    """Analyze model performance by subject."""
    subject_analysis = {}
    all_subjects = set()
    
    for model_name, data in results.items():
        for result in data['results']:
            if 'subject' in result:
                all_subjects.add(result['subject'])
    
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
    """Generate detailed markdown report."""
    os.makedirs(output_dir, exist_ok=True)
    report_file = os.path.join(output_dir, 'mmlu_pro_leaderboard_report.md')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# MMLU-PRO Subject-Specific Model Performance Leaderboard\n\n")
        f.write(f"**Evaluation Time**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Total Models**: {len(leaderboard)}\n\n")
        
        if leaderboard:
            f.write(f"**Top Accuracy**: {leaderboard[0]['Accuracy (%)']:.2f}% ({leaderboard[0]['Model']})\n\n")
            avg_accuracy = sum(item['Accuracy (%)'] for item in leaderboard) / len(leaderboard)
            f.write(f"**Average Accuracy**: {avg_accuracy:.2f}%\n\n")
        
        f.write("## Overall Leaderboard\n\n")
        f.write("| Rank | Model | Accuracy (%) | Correct | Total |\n")
        f.write("|------|--------|--------------|----------|--------|\n")
        for item in leaderboard:
            f.write(f"| {item['Rank']} | {item['Model']} | {item['Accuracy (%)']:.2f} | {item['Correct']} | {item['Total']} |\n")
        f.write("\n")
        
        f.write("## Detailed Per-Subject Analysis\n\n")
        for subject, model_scores in subject_analysis.items():
            f.write(f"### {subject}\n\n")
            f.write("| Model | Accuracy (%) | Correct | Total |\n")
            f.write("|--------|--------------|----------|--------|\n")
            sorted_models = sorted(model_scores.items(), key=lambda x: x[1]['accuracy'], reverse=True)
            for model_name, scores in sorted_models:
                f.write(f"| {model_name} | {scores['accuracy']:.2f} | {scores['correct']} | {scores['total']} |\n")
            f.write("\n")
    
    print(f"Detailed markdown report saved to: {report_file}")

def save_leaderboard_csv(leaderboard: List[Dict], output_dir: str):
    """Save leaderboard as CSV file."""
    import csv
    csv_file = os.path.join(output_dir, 'mmlu_pro_leaderboard.csv')
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        if leaderboard:
            writer = csv.DictWriter(f, fieldnames=leaderboard[0].keys())
            writer.writeheader()
            writer.writerows(leaderboard)
    
    print(f"Leaderboard CSV saved to: {csv_file}")

def main():
    base_dir = "/home/shared/RAG_DATA"
    output_dir = os.path.join(base_dir, "mmlu_pro_analysis_results")
    
    print("Loading MMLU-PRO evaluation results...")
    results = load_model_results(base_dir)
    
    if not results:
        print("No model results found. Please run the MMLU-PRO evaluation first.")
        return
    
    print(f"\nFound {len(results)} models with evaluation results.")
    
    print("\nGenerating model leaderboard...")
    leaderboard = generate_leaderboard(results)
    
    if not leaderboard:
        print("Failed to generate leaderboard.")
        return
    
    print("\n" + "=" * 80)
    print("MMLU-PRO Subject-Specific Model Leaderboard")
    print("=" * 80)
    print(f"{'Rank':<4} {'Model':<25} {'Accuracy(%)':<12} {'Correct':<8} {'Total':<6}")
    print("-" * 80)
    for item in leaderboard:
        print(f"{item['Rank']:<4} {item['Model']:<25} {item['Accuracy (%)']:<12.2f} {item['Correct']:<8} {item['Total']:<6}")
    print("=" * 80)
    
    print("\nAnalyzing performance by subject...")
    subject_analysis = analyze_by_subject(results)
    
    print("\nBest model per subject:")
    for subject, model_scores in subject_analysis.items():
        if model_scores:
            best_model = max(model_scores.items(), key=lambda x: x[1]['accuracy'])
            print(f"  {subject}: {best_model[0]} ({best_model[1]['accuracy']:.2f}%)")
    
    os.makedirs(output_dir, exist_ok=True)
    
    save_leaderboard_csv(leaderboard, output_dir)
    
    print("\nGenerating detailed markdown report...")
    generate_report(leaderboard, subject_analysis, output_dir)
    
    print(f"\nAnalysis completed. All results saved to: {output_dir}")

if __name__ == "__main__":
    main()
