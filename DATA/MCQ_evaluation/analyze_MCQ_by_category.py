#!/usr/bin/env python3
"""
Analyze MCQ results by intent and event categories.
This script analyzes testMCQ_set results by linking them to intent/event categories.
"""

import os
import json
from pathlib import Path
from collections import defaultdict

def load_query2category_mapping():
    """Load the mapping from original_query to type_category"""
    mapping_file = "/home/shared/RAG_DATA/benchmark/category/query2type_category.json"
    
    if not os.path.exists(mapping_file):
        print(f"❌ Mapping file not found: {mapping_file}")
        return {}
    
    print(f"📂 Loading query to category mapping from: {mapping_file}")
    
    with open(mapping_file, 'r', encoding='utf-8') as f:
        mapping_data = json.load(f)
    
    # Create lookup dictionary
    query2category = {}
    for item in mapping_data:
        original_query = item.get('original_query', '')
        type_category = item.get('type_category', '')
        if original_query and type_category:
            query2category[original_query] = type_category
    
    print(f"✅ Loaded {len(query2category)} query-category mappings")
    return query2category

def load_test_set_mapping():
    """Load the mapping from (question + options) to original_query"""
    test_set_file = "/home/shared/RAG_DATA/benchmark/MCQ/generated_test_sets/test_set_base_simple.json"
    
    if not os.path.exists(test_set_file):
        print(f"❌ Test set file not found: {test_set_file}")
        return {}
    
    print(f"📂 Loading test set mapping from: {test_set_file}")
    
    with open(test_set_file, 'r', encoding='utf-8') as f:
        test_set_data = json.load(f)
    
    # Create lookup dictionary
    question_options2query = {}
    for item in test_set_data:
        if 'multiple_choice' in item and 'gpt40' in item['multiple_choice']:
            content = item['multiple_choice']['gpt40']['content']
            question = content.get('question', '')
            options = content.get('options', [])
            original_query = item.get('original_query', '')
            
            if question and options and original_query:
                # Create key from question + options
                options_str = "\n".join(options)
                key = f"{question}\n{options_str}"
                question_options2query[key] = original_query
    
    print(f"✅ Loaded {len(question_options2query)} question-options to query mappings")
    return question_options2query

def load_model_results(model_dir, model_name):
    """Load results for a specific model"""
    results = {}
    
    # Check for different file naming patterns
    file_patterns = [
        ('base', ['base_test.json', 'base_results.json']),
        ('mix', ['mix_test.json', 'mix_results.json']),
        ('golden', ['golden_test.json', 'golden_results.json'])
    ]
    
    for dataset_type, filenames in file_patterns:
        for filename in filenames:
            file_path = os.path.join(model_dir, filename)
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    results[dataset_type] = data
                    print(f"  ✅ {dataset_type}: {len(data)} results loaded")
                except Exception as e:
                    print(f"  ❌ Error loading {dataset_type}: {e}")
                break
        else:
            print(f"  ❌ {dataset_type}: No results file found")
    
    return results

def extract_intent_event(type_category):
    """Extract intent and event from type_category string"""
    if not type_category or '_' not in type_category:
        return None, None
    
    # Split by underscore, intent is first part, event is second part
    parts = type_category.split('_', 1)
    if len(parts) >= 2:
        intent = parts[0]
        event = parts[1]
        return intent, event
    
    return None, None

def analyze_model_by_category(model_results, question_options2query, query2category):
    """Analyze model results by intent and event categories"""
    category_results = defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0}))
    
    for dataset_type, results_data in model_results.items():
        print(f"  📊 Analyzing {dataset_type} dataset...")
        
        for item in results_data:
            is_correct = item.get('is_correct', False)
            original_query = None
            
            # 优先使用original_query字段（如果存在）
            if 'original_query' in item:
                original_query = item['original_query']
            else:
                # 备用方案：通过question和options匹配
                question = item.get('question', '')
                options = item.get('options', [])
                
                if question and options:
                    options_str = "\n".join(options)
                    key = f"{question}\n{options_str}"
                    original_query = question_options2query.get(key)
            
            if original_query:
                # Find type_category
                type_category = query2category.get(original_query)
                if type_category:
                    # Extract intent and event
                    intent, event = extract_intent_event(type_category)
                    if intent and event:
                        # Update statistics
                        category_results[dataset_type][f"{intent}_{event}"]['total'] += 1
                        if is_correct:
                            category_results[dataset_type][f"{intent}_{event}"]['correct'] += 1
    
    return category_results

def calculate_accuracy_by_category(category_results):
    """Calculate accuracy for each category"""
    accuracy_results = {}
    
    for dataset_type, categories in category_results.items():
        accuracy_results[dataset_type] = {}
        
        for category, stats in categories.items():
            total = stats['total']
            correct = stats['correct']
            accuracy = correct / total if total > 0 else 0
            accuracy_results[dataset_type][category] = {
                'accuracy': accuracy,
                'correct': correct,
                'total': total
            }
    
    return accuracy_results

def aggregate_by_intent_event(accuracy_results):
    # accuracy_results: {category: {accuracy, correct, total}}
    intent_stats = {}
    event_stats = {}
    for category, stat in accuracy_results.items():
        if '_' not in category:
            continue
        intent, event = category.split('_', 1)
        # intent聚合
        if intent not in intent_stats:
            intent_stats[intent] = {'correct': 0, 'total': 0}
        intent_stats[intent]['correct'] += stat['correct']
        intent_stats[intent]['total'] += stat['total']
        # event聚合
        if event not in event_stats:
            event_stats[event] = {'correct': 0, 'total': 0}
        event_stats[event]['correct'] += stat['correct']
        event_stats[event]['total'] += stat['total']
    # 计算准确率
    intent_acc = {k: v['correct']/v['total'] if v['total'] else 0 for k, v in intent_stats.items()}
    event_acc = {k: v['correct']/v['total'] if v['total'] else 0 for k, v in event_stats.items()}
    return intent_acc, event_acc

def print_summary_table(all_model_results):
    """Print summary table of all results"""
    print(f"\n{'='*100}")
    print("📋 SUMMARY BY INTENT AND EVENT CATEGORIES")
    print("=" * 100)
    
    # Collect all unique categories
    all_categories = set()
    for model_name, model_data in all_model_results.items():
        for dataset_type, categories in model_data.items():
            all_categories.update(categories.keys())
    
    all_categories = sorted(all_categories)
    
    # Create summary data
    summary_data = []
    
    for model_name, model_data in all_model_results.items():
        for dataset_type in ['base', 'mix', 'golden']:
            if dataset_type in model_data:
                row = {'Model': model_name, 'Dataset': dataset_type}
                
                for category in all_categories:
                    if category in model_data[dataset_type]:
                        result = model_data[dataset_type][category]
                        row[f'{category}_acc'] = f"{result['accuracy']:.3f}"
                        row[f'{category}_count'] = f"{result['correct']}/{result['total']}"
                    else:
                        row[f'{category}_acc'] = "N/A"
                        row[f'{category}_count'] = "N/A"
                
                summary_data.append(row)
    
    # Print summary table using native Python
    print("\n📊 Summary Table:")
    print("="*100)
    
    # Print header
    header = ['Model', 'Dataset']
    for category in all_categories:
        header.extend([f'{category} Acc', f'{category} Count'])
    
    print(" | ".join(f"{col:<15}" for col in header))
    print("-" * (len(header) * 18))
    
    # Print data rows
    for row in summary_data:
        display_row = [row['Model'], row['Dataset']]
        for category in all_categories:
            acc_key = f'{category}_acc'
            count_key = f'{category}_count'
            display_row.extend([
                str(row.get(acc_key, "N/A")), 
                str(row.get(count_key, "N/A"))
            ])
        
        print(" | ".join(f"{str(val):<15}" for val in display_row))
    
    return summary_data

def save_detailed_results(all_model_results, output_file):
    """Save detailed results to JSON file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_model_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Detailed results saved to: {output_file}")

def main():
    print("🔍 Analyzing MCQ results by intent and event categories...")
    print("=" * 80)
    
    # Load mappings
    query2category = load_query2category_mapping()
    question_options2query = load_test_set_mapping()
    
    if not query2category or not question_options2query:
        print("❌ Failed to load required mappings")
        return
    
    # Get testMCQ_set directory
    testMCQ_dir = "/home/shared/RAG_DATA/DATA/local_MCQ"
    
    if not os.path.exists(testMCQ_dir):
        print(f"❌ Directory {testMCQ_dir} not found")
        return
    
    # Get all model directories
    model_dirs = [d for d in os.listdir(testMCQ_dir) if os.path.isdir(os.path.join(testMCQ_dir, d))]
    model_dirs.sort()
    
    all_model_results = {}
    
    print(f"\n📊 Processing {len(model_dirs)} models...")
    
    for model_name in model_dirs:
        model_dir = os.path.join(testMCQ_dir, model_name)
        print(f"\n🤖 {model_name}:")
        print("-" * 40)
        
        # Load model results
        model_results = load_model_results(model_dir, model_name)
        
        if model_results:
            # Analyze by category
            category_results = analyze_model_by_category(model_results, question_options2query, query2category)
            
            # Calculate accuracy
            accuracy_results = calculate_accuracy_by_category(category_results)
            
            all_model_results[model_name] = accuracy_results
            
            # Print model-specific summary
            print(f"  📈 Category breakdown:")
            for dataset_type, categories in accuracy_results.items():
                print(f"    {dataset_type.upper()}:")
                for category, result in categories.items():
                    print(f"      {category}: {result['accuracy']:.3f} ({result['correct']}/{result['total']})")
    
    # Print overall summary table
    summary_data = print_summary_table(all_model_results)
    
    # Save detailed results
    output_file = "/home/shared/RAG_DATA/DATA/MCQ_category_analysis_results.json"
    save_detailed_results(all_model_results, output_file)
    

    
    # Aggregate by intent and event
    aggregated_results = {}
    for model_name, model_data in all_model_results.items():
        aggregated_results[model_name] = {}
        for dataset_type, categories in model_data.items():
            aggregated_results[model_name][dataset_type] = {}
            for category, stat in categories.items():
                aggregated_results[model_name][dataset_type][category] = stat

    # Calculate aggregated accuracies
    intent_acc, event_acc = aggregate_by_intent_event(aggregated_results)

    print("\n📊 Aggregated Accuracy by Intent and Event:")
    for intent, acc in intent_acc.items():
        print(f"  {intent}: {acc:.3f}")
    for event, acc in event_acc.items():
        print(f"  {event}: {acc:.3f}")

    print(f"\n🎉 Analysis completed!")
    print(f"📁 Results saved in: /home/shared/RAG_DATA/DATA/")

if __name__ == "__main__":
    main() 