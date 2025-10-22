#!/usr/bin/env python3
"""
Extract key points using GPT-4o-mini, forcing the count to match golden standard
"""

import json
import asyncio
import openai
import os
from typing import List, Dict, Any
import argparse
from dotenv import load_dotenv
from tqdm import tqdm

# Configure OpenAI API
def get_openai_client():
    # Load .env file in DATA directory
    load_dotenv('DATA/.env')
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in DATA/.env file")
        print("Please add OPENAI_API_KEY=your-api-key-here to DATA/.env file")
        return None
    return openai.AsyncOpenAI(api_key=api_key)

async def extract_keypoints_with_gpt4o_mini(question: str, model_answer: str, golden_keypoints: List[str]) -> List[str]:
    """
    Extract key points using GPT-4o-mini, forcing the count to match golden standard
    """
    client = get_openai_client()
    if not client:
        return ["API key not configured"] * len(golden_keypoints)
    
    target_count = len(golden_keypoints)
    
    prompt = f"""Please extract {target_count} key points from the following model answer. Each key point should be a complete sentence that covers important information from the model answer.

Question: {question}

Model Answer: {model_answer}

Please extract exactly {target_count} key points in the following format:
1. [Key point 1]
2. [Key point 2]
...
{target_count}. [Key point {target_count}]

Return only the key points list, no other content."""

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1000
        )
        
        content = response.choices[0].message.content.strip()
        
        # Parse returned key points
        keypoints = []
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('•') or line.startswith('-')):
                # Remove numbering and symbols
                if '. ' in line:
                    keypoint = line.split('. ', 1)[1]
                elif '• ' in line:
                    keypoint = line[2:]
                elif '- ' in line:
                    keypoint = line[2:]
                else:
                    keypoint = line
                
                if keypoint:
                    keypoints.append(keypoint)
        
        # Adjust if the number of extracted key points doesn't match
        if len(keypoints) > target_count:
            keypoints = keypoints[:target_count]
        elif len(keypoints) < target_count:
            # If not enough, fill with the last key point
            while len(keypoints) < target_count:
                keypoints.append(keypoints[-1] if keypoints else "No key point available")
        
        return keypoints
        
    except Exception as e:
        print(f"Error extracting keypoints: {e}")
        # Return default key points
        return ["Error in keypoint extraction"] * target_count

async def process_model_results(model_name: str, golden_data: List[Dict], concurrency: int = 3000, test_mode: bool = False, test_count: int = 5, test_set: str = "base"):
    """
    Process results for a single model
    """
    # Read model results
    model_results_path = f"DATA/local_OE/{model_name}/{test_set}_oe_with_difficulty.json"
    if not os.path.exists(model_results_path):
        print(f"❌ Model results not found: {model_results_path}")
        return
    
    with open(model_results_path, 'r', encoding='utf-8') as f:
        model_results = json.load(f)
    
    # Check if it's a mix test set, if so use golden lookup
    is_mix = test_set == "mix" and any('passage' in item and item['passage'] for item in model_results)
    golden_lookup = {}
    if is_mix:
        print(f"🔍 Detected mix test set for {model_name}, using golden lookup method")
        # Load golden results as lookup
        golden_path = f"DATA/local_OE/{model_name}/golden_oe_with_difficulty.json"
        if os.path.exists(golden_path):
            with open(golden_path, 'r', encoding='utf-8') as f:
                golden_data_mix = json.load(f)
            
            for item in golden_data_mix:
                question = item['question']
                passage = item.get('passage', '')
                key = f"{question}_{passage}"
                golden_lookup[key] = {
                    'model_answer': item['model_answer'],
                    'model_keypoint': item.get('model_keypoint', None),
                    'answer_coverage': item.get('answer_coverage', None)
                }
            print(f"📊 Loaded {len(golden_lookup)} golden results for lookup")
        else:
            print(f"⚠️  Golden results not found at {golden_path}")
            golden_lookup = {}
    
    # Test mode: process only the first few items
    if test_mode:
        model_results = model_results[:test_count]
        print(f"🧪 TEST MODE: Processing only {test_count} items")
    
    # Create lookup dictionary for golden data
    golden_lookup = {}
    for item in golden_data:
        # Get question from open_ended.gpt40.content.question
        question = item.get('open_ended', {}).get('gpt40', {}).get('content', {}).get('question', '')
        if question:
            golden_lookup[question] = item
    
    # Create semaphore to control concurrency
    semaphore = asyncio.Semaphore(concurrency)
    
    async def process_single_item(item):
        async with semaphore:
            question = item['question']
            model_answer = item['model_answer']
            passage = item.get('passage', '')
            
            # For mix test set, check golden lookup first
            if is_mix:
                lookup_key = f"{question}_{passage}"
                if lookup_key in golden_lookup:
                    golden_result = golden_lookup[lookup_key]
                    # Directly copy key points from golden result
                    item['model_keypoint'] = golden_result.get('model_keypoint', None)
                    item['answer_coverage'] = golden_result.get('answer_coverage', None)
                    return True
            
            # Find corresponding golden data
            if question in golden_lookup:
                golden_item = golden_lookup[question]
                golden_keypoints = golden_item.get('open_ended', {}).get('gpt40', {}).get('content', {}).get('key_points', [])
                
                # If key_points not found, try other paths
                if not golden_keypoints:
                    # Try to get directly from open_ended
                    open_ended = golden_item.get('open_ended', {})
                    for model_key in open_ended:
                        content = open_ended[model_key].get('content', {})
                        golden_keypoints = content.get('key_points', [])
                        if golden_keypoints:
                            break
                
                if golden_keypoints:
                    extracted_keypoints = await extract_keypoints_with_gpt4o_mini(question, model_answer, golden_keypoints)
                    
                    # Add model_keypoint field, answer_coverage remains unchanged
                    # Ensure field order: model_answer -> model_keypoint -> answer_coverage
                    item['model_keypoint'] = extracted_keypoints
                    return True
            
            return False
    
        # Process all items with real-time progress
    total_items = len(model_results)
    print(f"Starting processing {total_items} items for {model_name}...")
    
    # Create progress bar
    pbar = tqdm(total=total_items, desc=f"Processing {model_name}", unit="items")
    
    # Track progress
    completed_count = 0
    updated_count = 0
    
    async def process_with_progress(item):
        nonlocal completed_count, updated_count
        result = await process_single_item(item)
        completed_count += 1
        if result:
            updated_count += 1
        pbar.update(1)
        pbar.set_postfix({
            'Completed': f"{completed_count}/{total_items}",
            'Updated': updated_count,
            'Success Rate': f"{updated_count/completed_count*100:.1f}%" if completed_count > 0 else "0%"
        })
        return result
    
    # Process all items concurrently
    tasks = [process_with_progress(item) for item in model_results]
    results = await asyncio.gather(*tasks)
    
    pbar.close()
    
    # Final summary
    print(f"✅ Updated {updated_count}/{total_items} items for {model_name} ({updated_count/total_items*100:.1f}%)")
    
    # Ensure correct field order: question, correct_answer, model_answer, model_keypoint, answer_coverage, difficulty, passage
    for item in model_results:
        if 'model_keypoint' in item:
            # Reorder fields to ensure model_keypoint is between model_answer and answer_coverage
            ordered_item = {
                'question': item['question'],
                'correct_answer': item['correct_answer'],
                'model_answer': item['model_answer'],
                'model_keypoint': item['model_keypoint'],
                'answer_coverage': item.get('answer_coverage', None),
                'difficulty': item['difficulty']
            }
            if 'passage' in item:
                ordered_item['passage'] = item['passage']
            # Replace the original item with ordered item
            item.clear()
            item.update(ordered_item)
    
    if test_mode:
        # Test mode: save to test file
        test_output_path = f"DATA/local_OE/{model_name}/{test_set}_oe_with_difficulty_test.json"
        with open(test_output_path, 'w', encoding='utf-8') as f:
            json.dump(model_results, f, indent=2, ensure_ascii=False)
        print(f"🧪 Test results saved to: {test_output_path}")
    else:
        # Normal mode: save to original file
        with open(model_results_path, 'w', encoding='utf-8') as f:
            json.dump(model_results, f, indent=2, ensure_ascii=False)

async def main():
    parser = argparse.ArgumentParser(description='Extract keypoints using GPT-4o-mini')
    parser.add_argument('--model', type=str, help='Model name to process (e.g., gpt-4o, gemini-1.5-pro)')
    parser.add_argument('--test_set', type=str, default='base', choices=['base', 'golden', 'mix'], help='Test set to process')
    parser.add_argument('--final', action='store_true', help='Process all test sets (base, golden, mix) for the model')
    parser.add_argument('--concurrency', type=int, default=3000, help='Concurrency level')
    parser.add_argument('--test_mode', action='store_true', help='Test mode: process only 5 items')
    parser.add_argument('--test_count', type=int, default=5, help='Number of items to process in test mode')
    args = parser.parse_args()
    
    # Read golden data
    print("Loading golden data...")
    with open('DATA/final_OE/golden_oe_with_difficulty.json', 'r', encoding='utf-8') as f:
        golden_data = json.load(f)
    
    if args.model:
        if args.final:
            # Process all test sets for the specified model
            test_sets = ['base', 'golden', 'mix']
            for test_set in test_sets:
                print(f"\n{'='*50}")
                print(f"Processing {args.model} - {test_set} test set")
                print(f"{'='*50}")
                await process_model_results(args.model, golden_data, args.concurrency, args.test_mode, args.test_count, test_set)
        else:
            # Process single test set for the specified model
            await process_model_results(args.model, golden_data, args.concurrency, args.test_mode, args.test_count, args.test_set)
    else:
        # Process all models
        models_to_process = [
            "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo",
            "gemini-2.5-pro", "gemini-2.5-flash", "gemini-1.5-pro", "gemini-1.5-flash"
        ]
        
        for model in models_to_process:
            print(f"\nProcessing model: {model}")
            await process_model_results(model, golden_data, args.concurrency, args.test_mode, args.test_count, args.test_set)

if __name__ == "__main__":
    asyncio.run(main())
