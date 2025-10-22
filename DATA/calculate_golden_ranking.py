#!/usr/bin/env python3
"""
Calculate golden version accuracy ranking for all models in local_MCQ
"""

import json
import os
from collections import defaultdict

def load_mcq_summary():
    """Load MCQ summary data"""
    summary_file = "/home/shared/RAG_DATA/DATA/MCQ_summary_by_intent_event.json"
    
    if not os.path.exists(summary_file):
        print(f"❌ Summary file not found: {summary_file}")
        return None
    
    with open(summary_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data

def get_existing_models():
    """Get list of models that actually exist in local_MCQ directory"""
    mcq_dir = "/home/shared/RAG_DATA/DATA/local_MCQ"
    if not os.path.exists(mcq_dir):
        print(f"❌ MCQ directory not found: {mcq_dir}")
        return set()
    
    existing_models = set()
    for item in os.listdir(mcq_dir):
        if os.path.isdir(os.path.join(mcq_dir, item)):
            existing_models.add(item)
    
    print(f"📂 Found {len(existing_models)} models in local_MCQ directory")
    return existing_models

def calculate_golden_average_accuracy(summary_data):
    """Calculate average accuracy across all intents for golden dataset"""
    if 'golden' not in summary_data or 'intents' not in summary_data['golden']:
        print("❌ Golden data not found in summary")
        return {}
    
    # Get existing models
    existing_models = get_existing_models()
    if not existing_models:
        return {}
    
    golden_intents = summary_data['golden']['intents']
    model_accuracies = defaultdict(list)
    
    # Collect all accuracies for each model across all intents
    for intent, models in golden_intents.items():
        for model_name, accuracy in models.items():
            # Only include models that actually exist in local_MCQ
            if model_name in existing_models:
                model_accuracies[model_name].append(accuracy)
    
    # Calculate average for each model
    average_accuracies = {}
    for model_name, accuracies in model_accuracies.items():
        avg_accuracy = sum(accuracies) / len(accuracies)
        average_accuracies[model_name] = avg_accuracy
    
    print(f"📊 Calculated accuracies for {len(average_accuracies)} existing models")
    return average_accuracies

def get_model_info():
    """Get model information including parameter count"""
    model_info = {
        "Yi-6B-Chat": {"params": "6B", "org": "01.AI"},
        "Hunyuan-7B-Instruct": {"params": "7B", "org": "腾讯"},
        "Hunyuan-0.5B-Instruct": {"params": "0.5B", "org": "腾讯"},
        "Hunyuan-4B-Instruct": {"params": "4B", "org": "腾讯"},
        "Falcon3-1B-Instruct": {"params": "1B", "org": "Technology Innovation Institute"},
        "AceMath-1.5B-Instruct": {"params": "1.5B", "org": "NVIDIA"},
        "llama-3.2-3b-instruct": {"params": "3.21B", "org": "Meta"},
        "qwen-3-0.6b": {"params": "0.6B", "org": "阿里云"},
        "Llama-3.2-1B-Instruct": {"params": "1.23B", "org": "Meta"},
        "phi-2": {"params": "2.7B", "org": "Microsoft"},
        "qwen-2.5-3b-instruct": {"params": "3.09B", "org": "阿里云"},
        "qwen-3-4b": {"params": "4B", "org": "阿里云"},
        "llama-3-8b": {"params": "8B", "org": "Meta"},
        "gemma-7b": {"params": "7B", "org": "Google"},
        "deepseek-v3-7b": {"params": "7B", "org": "DeepSeek"},
        "qwen-3-8b": {"params": "8B", "org": "阿里云"},
        "Mistral-7B-Instruct-v0.2": {"params": "7B", "org": "Mistral AI"},
        "gemini-1.5-pro": {"params": "Unknown", "org": "Google"},
        "gemini-2.5-pro": {"params": "Unknown", "org": "Google"},
        "gpt-4o": {"params": "Unknown", "org": "OpenAI"}
    }
    return model_info

def print_ranking(average_accuracies, model_info):
    """Print the ranking table"""
    print("🏆 GOLDEN VERSION ACCURACY RANKING")
    print("=" * 80)
    print(f"{'Rank':<4} {'Model':<25} {'Params':<8} {'Org':<20} {'Avg Accuracy':<12}")
    print("-" * 80)
    
    # Sort by accuracy (descending)
    sorted_models = sorted(average_accuracies.items(), key=lambda x: x[1], reverse=True)
    
    for rank, (model_name, accuracy) in enumerate(sorted_models, 1):
        info = model_info.get(model_name, {"params": "Unknown", "org": "Unknown"})
        print(f"{rank:<4} {model_name:<25} {info['params']:<8} {info['org']:<20} {accuracy:.4f}")
    
    print("-" * 80)
    print(f"Total models ranked: {len(sorted_models)}")

def save_ranking_to_file(average_accuracies, model_info):
    """Save ranking to JSON file"""
    ranking_data = []
    model_info_dict = get_model_info()
    
    # Sort by accuracy (descending)
    sorted_models = sorted(average_accuracies.items(), key=lambda x: x[1], reverse=True)
    
    for rank, (model_name, accuracy) in enumerate(sorted_models, 1):
        info = model_info_dict.get(model_name, {"params": "Unknown", "org": "Unknown"})
        ranking_data.append({
            "rank": rank,
            "model_name": model_name,
            "parameters": info["params"],
            "organization": info["org"],
            "average_accuracy": accuracy
        })
    
    output_file = "/home/shared/RAG_DATA/DATA/golden_mcq_ranking.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(ranking_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Ranking saved to: {output_file}")

def main():
    print("🔍 Calculating golden version accuracy ranking...")
    print("=" * 60)
    
    # Load summary data
    summary_data = load_mcq_summary()
    if not summary_data:
        return
    
    # Calculate average accuracies
    average_accuracies = calculate_golden_average_accuracy(summary_data)
    if not average_accuracies:
        return
    
    # Get model information
    model_info = get_model_info()
    
    # Print ranking
    print_ranking(average_accuracies, model_info)
    
    # Save to file
    save_ranking_to_file(average_accuracies, model_info)
    
    print(f"\n🎉 Ranking calculation completed!")

if __name__ == "__main__":
    main()
