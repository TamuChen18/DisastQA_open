#!/usr/bin/env python3
"""
Summarize MCQ results by intent and event categories.
This script creates a summary based on the existing MCQ_category_analysis_results.json
"""

import os
import json
import pandas as pd
from collections import defaultdict

def load_existing_results():
    """Load the existing MCQ category analysis results"""
    results_file = "/home/shared/RAG_DATA/benchmark/MCQ_category_analysis_results.json"
    
    if not os.path.exists(results_file):
        print(f"❌ Results file not found: {results_file}")
        return {}
    
    print(f"📂 Loading existing results from: {results_file}")
    
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    print(f"✅ Loaded results for {len(results)} models")
    return results

def aggregate_by_intent_event(category_results):
    """Aggregate results by intent and event separately"""
    intent_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    event_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for category, stats in category_results.items():
        if '_' not in category:
            continue
        
        # Split by first underscore
        parts = category.split('_', 1)
        if len(parts) >= 2:
            intent = parts[0]
            event = parts[1]
            
            # Aggregate by intent
            intent_stats[intent]['correct'] += stats['correct']
            intent_stats[intent]['total'] += stats['total']
            
            # Aggregate by event
            event_stats[event]['correct'] += stats['correct']
            event_stats[event]['total'] += stats['total']
    
    # Calculate accuracies
    intent_accuracies = {}
    for intent, stats in intent_stats.items():
        if stats['total'] > 0:
            intent_accuracies[intent] = stats['correct'] / stats['total']
    
    event_accuracies = {}
    for event, stats in event_stats.items():
        if stats['total'] > 0:
            event_accuracies[event] = stats['correct'] / stats['total']
    
    return intent_accuracies, event_accuracies

def create_summary_table(all_results):
    """Create a summary table with intent and event aggregations"""
    summary_data = []
    
    # Collect all unique intents and events
    all_intents = set()
    all_events = set()
    
    for model_name, model_data in all_results.items():
        for dataset_type, categories in model_data.items():
            intent_acc, event_acc = aggregate_by_intent_event(categories)
            all_intents.update(intent_acc.keys())
            all_events.update(event_acc.keys())
    
    all_intents = sorted(all_intents)
    all_events = sorted(all_events)
    
    print(f"📊 Found {len(all_intents)} intents: {all_intents}")
    print(f"📊 Found {len(all_events)} events: {all_events}")
    
    # Create summary rows
    for model_name, model_data in all_results.items():
        for dataset_type in ['base', 'mix', 'golden']:
            if dataset_type in model_data:
                row = {
                    'Model': model_name,
                    'Dataset': dataset_type
                }
                
                # Add intent accuracies
                intent_acc, _ = aggregate_by_intent_event(model_data[dataset_type])
                for intent in all_intents:
                    row[f'{intent}_acc'] = f"{intent_acc.get(intent, 0):.3f}"
                
                # Add event accuracies
                _, event_acc = aggregate_by_intent_event(model_data[dataset_type])
                for event in all_events:
                    row[f'{event}_acc'] = f"{event_acc.get(event, 0):.3f}"
                
                summary_data.append(row)
    
    return summary_data, all_intents, all_events

def print_summary_by_dataset(all_results):
    """Print summary organized by dataset (base/mix/golden)"""
    print(f"\n{'='*120}")
    print("📋 SUMMARY BY DATASET")
    print("=" * 120)
    
    for dataset_type in ['base', 'mix', 'golden']:
        print(f"\n🎯 {dataset_type.upper()} DATASET:")
        print("-" * 80)
        
        # Collect all intents and events for this dataset
        all_intents = set()
        all_events = set()
        
        for model_name, model_data in all_results.items():
            if dataset_type in model_data:
                intent_acc, event_acc = aggregate_by_intent_event(model_data[dataset_type])
                all_intents.update(intent_acc.keys())
                all_events.update(event_acc.keys())
        
        all_intents = sorted(all_intents)
        all_events = sorted(all_events)
        
        # Print intent summary
        print(f"\n📊 INTENT ACCURACIES:")
        intent_header = f"{'Model':<20}"
        for intent in all_intents:
            intent_header += f" {intent:<10}"
        print(intent_header)
        print("-" * len(intent_header))
        
        for model_name, model_data in all_results.items():
            if dataset_type in model_data:
                intent_acc, _ = aggregate_by_intent_event(model_data[dataset_type])
                row = f"{model_name:<20}"
                for intent in all_intents:
                    acc = intent_acc.get(intent, 0)
                    row += f" {acc:.3f}     "
                print(row)
        
        # Print event summary
        print(f"\n📊 EVENT ACCURACIES:")
        event_header = f"{'Model':<20}"
        for event in all_events:
            event_header += f" {event:<10}"
        print(event_header)
        print("-" * len(event_header))
        
        for model_name, model_data in all_results.items():
            if dataset_type in model_data:
                _, event_acc = aggregate_by_intent_event(model_data[dataset_type])
                row = f"{model_name:<20}"
                for event in all_events:
                    acc = event_acc.get(event, 0)
                    row += f" {acc:.3f}     "
                print(row)

def save_summary_results(summary_data, all_intents, all_events, all_results):
    """Save summary results to files"""
    # Create DataFrame
    df = pd.DataFrame(summary_data)
    
    # Reorder columns for better readability
    base_columns = ['Model', 'Dataset']
    intent_columns = [f'{intent}_acc' for intent in all_intents]
    event_columns = [f'{event}_acc' for event in all_events]
    
    df = df[base_columns + intent_columns + event_columns]
    
    # Rename columns for display
    new_columns = ['Model', 'Dataset']
    new_columns.extend([f'{intent} Acc' for intent in all_intents])
    new_columns.extend([f'{event} Acc' for event in all_events])
    
    df.columns = new_columns
    
    # Save to CSV
    csv_file = "/home/shared/RAG_DATA/benchmark/MCQ_summary_by_intent_event.csv"
    df.to_csv(csv_file, index=False)
    print(f"✅ Summary table saved to: {csv_file}")
    
    # Save to JSON
    json_file = "/home/shared/RAG_DATA/benchmark/MCQ_summary_by_intent_event.json"
    
    # Convert to structured JSON
    summary_json = {}
    for dataset_type in ['base', 'mix', 'golden']:
        summary_json[dataset_type] = {
            'intents': {},
            'events': {}
        }
        
        # Collect data for this dataset
        for model_name, model_data in all_results.items():
            if dataset_type in model_data:
                intent_acc, event_acc = aggregate_by_intent_event(model_data[dataset_type])
                
                # Add to intents
                for intent, acc in intent_acc.items():
                    if intent not in summary_json[dataset_type]['intents']:
                        summary_json[dataset_type]['intents'][intent] = {}
                    summary_json[dataset_type]['intents'][intent][model_name] = acc
                
                # Add to events
                for event, acc in event_acc.items():
                    if event not in summary_json[dataset_type]['events']:
                        summary_json[dataset_type]['events'][event] = {}
                    summary_json[dataset_type]['events'][event][model_name] = acc
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(summary_json, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Summary JSON saved to: {json_file}")
    
    return df

def main():
    print("🔍 Creating summary of MCQ results by intent and event...")
    print("=" * 80)
    
    # Load existing results
    all_results = load_existing_results()
    
    if not all_results:
        print("❌ No results to summarize")
        return
    
    # Create summary table
    summary_data, all_intents, all_events = create_summary_table(all_results)
    
    # Print summary by dataset
    print_summary_by_dataset(all_results)
    
    # Save results
    summary_df = save_summary_results(summary_data, all_intents, all_events, all_results)
    
    print(f"\n🎉 Summary completed!")
    print(f"📁 Results saved in: /home/shared/RAG_DATA/benchmark/")

if __name__ == "__main__":
    main() 