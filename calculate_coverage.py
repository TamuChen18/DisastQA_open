#!/usr/bin/env python3
import json
import argparse
import asyncio
import aiohttp
from typing import List, Dict, Tuple
from pathlib import Path
import os
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv('DATA/.env')

def get_openai_client():
    """Get OpenAI client"""
    from openai import AsyncOpenAI
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    return AsyncOpenAI(api_key=api_key)

async def analyze_coverage_with_gpt4o(question: str, golden_keypoints: List[str], 
                                     model_keypoints: List[str], client) -> Dict:
    """
    Use GPT-4o to analyze coverage between golden and model keypoints
    """
    if not golden_keypoints:
        return {
            'coverage': 0.0,
            'golden_count': 0,
            'model_count': len(model_keypoints),
            'matches': 0,
            'analysis': "No golden keypoints provided"
        }
    
    # Create prompt for GPT-4o analysis
    golden_text = "\n".join([f"{i+1}. {kp}" for i, kp in enumerate(golden_keypoints)])
    model_text = "\n".join([f"{i+1}. {kp}" for i, kp in enumerate(model_keypoints)]) if model_keypoints else "No keypoints provided"
    
    prompt = f"""Please analyze the coverage of model keypoints against the golden standard keypoints for the following question.

Question: {question}

Golden Standard Keypoints:
{golden_text}

Model Keypoints:
{model_text}

Please analyze which golden keypoints are covered by the model keypoints. Consider semantic similarity and information coverage, not just exact text matching.

For each golden keypoint, determine if it is covered by any model keypoint (either fully or partially). A model keypoint can cover multiple golden keypoints, and multiple model keypoints can cover the same golden keypoint.

Calculate the coverage percentage as: (number of covered golden keypoints / total number of golden keypoints) * 100

For example:
- If there are 3 golden keypoints and 2 are covered: coverage = (2/3) * 100 = 66.67%
- If there are 4 golden keypoints and 3 are covered: coverage = (3/4) * 100 = 75.00%
- If there are 2 golden keypoints and 0 are covered: coverage = (0/2) * 100 = 0.00%

Please respond in the following JSON format:
{{
    "covered_golden_indices": [list of indices (0-based) of golden keypoints that are covered],
    "coverage_percentage": float (calculated coverage percentage),
    "analysis": "brief explanation of which golden keypoints are covered and why"
}}

Return only the JSON response, no other text."""

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Parse JSON response
        try:
            result = json.loads(result_text)
            covered_count = len(result.get('covered_golden_indices', []))
            coverage = result.get('coverage_percentage', 0.0)
            analysis = result.get('analysis', '')
            
            return {
                'coverage': round(coverage, 2),
                'golden_count': len(golden_keypoints),
                'model_count': len(model_keypoints),
                'matches': covered_count,
                'analysis': analysis,
                'covered_indices': result.get('covered_golden_indices', [])
            }
        except json.JSONDecodeError:
            print(f"⚠️  Failed to parse GPT response: {result_text}")
            return {
                'coverage': 0.0,
                'golden_count': len(golden_keypoints),
                'model_count': len(model_keypoints),
                'matches': 0,
                'analysis': f"Failed to parse response: {result_text}"
            }
            
    except Exception as e:
        print(f"⚠️  Error calling GPT-4o: {e}")
        return {
            'coverage': 0.0,
            'golden_count': len(golden_keypoints),
            'model_count': len(model_keypoints),
            'matches': 0,
            'analysis': f"Error: {str(e)}"
        }

async def process_file_async(file_path: str, client, concurrency: int = 3000, dry_run: bool = False) -> Dict:
    """Process a single file and calculate coverage using GPT-4o"""
    print(f"Processing {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Check if coverage is already calculated
    sample_item = data[0] if data else {}
    if sample_item.get('answer_coverage') is not None and not isinstance(sample_item['answer_coverage'], str):
        print(f"⚠️  Coverage already calculated for {file_path}, skipping...")
        return {
            'total_items': len(data),
            'average_coverage': sum(item.get('answer_coverage', 0) for item in data) / len(data) if data else 0.0,
            'coverage_stats': []
        }
    
    total_coverage = 0.0
    total_items = 0
    coverage_stats = []
    
    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(concurrency)
    
    async def process_item(item):
        async with semaphore:
            return await analyze_coverage_with_gpt4o(
                item.get('question', ''),
                item.get('golden_keypoint', []),
                item.get('model_keypoint', []),
                client
            )
    
    # Process items with progress bar
    tasks = []
    for item in data:
        task = process_item(item)
        tasks.append(task)
    
    print(f"Processing {len(tasks)} items with concurrency {concurrency}...")
    results = []
    
    for i, task in enumerate(tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Analyzing coverage")):
        result = await task
        results.append(result)
        
        if not dry_run:
            # Add coverage info to the item
            data[i]['answer_coverage'] = result['coverage']
        
        total_coverage += result['coverage']
        total_items += 1
        coverage_stats.append(result)
    
    avg_coverage = total_coverage / total_items if total_items > 0 else 0.0
    
    if not dry_run:
        # Write back to file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✅ Updated {total_items} items, average coverage: {avg_coverage:.2f}%")
    else:
        print(f"🔍 Dry run: {total_items} items, average coverage: {avg_coverage:.2f}%")
    
    return {
        'total_items': total_items,
        'average_coverage': avg_coverage,
        'coverage_stats': coverage_stats
    }

async def process_model_results_async(model_name: str, concurrency: int = 3000, dry_run: bool = False):
    """Process all result files for a specific model"""
    base_dir = Path("DATA/final_OE/Final_Human_Review")
    model_dir = base_dir / model_name
    
    if not model_dir.exists():
        print(f"❌ Model directory not found: {model_dir}")
        return
    
    # Get OpenAI client
    client = get_openai_client()
    
    results = {}
    
    # Process base, golden, mix files
    for test_set in ['base', 'golden', 'mix']:
        file_path = model_dir / f"{test_set}_oe_with_difficulty.json"
        
        if file_path.exists():
            result = await process_file_async(str(file_path), client, concurrency, dry_run)
            results[test_set] = result
        else:
            print(f"⚠️  File not found: {file_path}")
    
    # Print summary
    print(f"\n📊 Coverage Summary for {model_name}:")
    print(f"{'Test Set':<10} {'Items':<8} {'Avg Coverage':<12} {'Golden Count':<12} {'Model Count':<12}")
    print("-" * 60)
    
    for test_set, result in results.items():
        avg_golden = sum(stat['golden_count'] for stat in result['coverage_stats']) / result['total_items']
        avg_model = sum(stat['model_count'] for stat in result['coverage_stats']) / result['total_items']
        print(f"{test_set:<10} {result['total_items']:<8} {result['average_coverage']:<12.2f} {avg_golden:<12.1f} {avg_model:<12.1f}")

async def main_async():
    parser = argparse.ArgumentParser(description="Calculate coverage between model and golden keypoints using GPT-4o")
    parser.add_argument("--model", type=str, help="Model name to process")
    parser.add_argument("--all", action="store_true", help="Process all models")
    parser.add_argument("--dry_run", action="store_true", help="Dry run without writing files")
    parser.add_argument("--concurrency", type=int, default=3000, help="Concurrency for API calls (default: 3000)")
    parser.add_argument("--test_count", type=int, help="Number of items to test (for testing)")
    
    args = parser.parse_args()
    
    if not args.model and not args.all:
        print("❌ Please specify --model or --all")
        return
    
    print(f"Using concurrency: {args.concurrency}")
    
    if args.all:
        # Process all models
        base_dir = Path("DATA/final_OE/Final_Human_Review")
        if not base_dir.exists():
            print(f"❌ Directory not found: {base_dir}")
            return
        
        model_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
        print(f"Found {len(model_dirs)} models to process")
        
        for model_dir in model_dirs:
            model_name = model_dir.name
            print(f"\n{'='*60}")
            print(f"Processing model: {model_name}")
            print(f"{'='*60}")
            await process_model_results_async(model_name, args.concurrency, args.dry_run)
    else:
        # Process specific model
        await process_model_results_async(args.model, args.concurrency, args.dry_run)

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
