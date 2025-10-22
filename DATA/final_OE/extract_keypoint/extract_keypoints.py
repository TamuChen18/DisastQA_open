#!/usr/bin/env python3
"""
Dynamic Key Points Extraction Script
Extract key points from correct_answer in DATA/final_OE/golden_oe.json
"""

import json
from openai import OpenAI
import time
import asyncio
import aiohttp
from typing import List, Dict, Any
import os
from dotenv import load_dotenv
import re
from concurrent.futures import ThreadPoolExecutor

# Load environment variables from DATA/.env file
load_dotenv('DATA/.env')

# Configure OpenAI API
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def clean_json_response(content: str) -> str:
    """
    Clean the response content to extract valid JSON
    """
    # Remove markdown code blocks
    content = re.sub(r'```json\s*', '', content)
    content = re.sub(r'```\s*$', '', content)
    
    # Remove any leading/trailing whitespace
    content = content.strip()
    
    return content

def extract_keypoints_with_gpt4o(answer: str, question: str = None) -> List[str]:
    """
    Extract key points from answer using GPT-4o
    """
    
    # Build prompt in English, letting GPT analyze and decide
    prompt = f"""Please analyze the following answer and extract key points that express the core ideas. 

Your task is to:
1. Analyze the answer content and complexity
2. Determine the appropriate number of key points needed based on content (no fixed limit)
3. Extract key points that capture the essential information and main arguments
4. Order key points by logical importance
5. Ensure key points are comprehensive enough to understand the main ideas

Guidelines:
- Let the content guide the number of key points (typically 1-10, but can be more if needed)
- For very short/simple answers: 1-2 key points may be sufficient
- For complex/detailed answers: 3-8 key points are usually appropriate
- For very comprehensive answers: 8+ key points may be needed
- Each key point should be a complete, independent idea
- Focus on expressing core concepts and main arguments
- Use concise but complete expressions
- Avoid repetition or redundancy
- Key points must be directly derived from the answer content

{f'Question: {question}' if question else ''}

Answer:
{answer}

Please analyze the answer and return key points in JSON format:
{{
    "key_points": [
        "Key point 1",
        "Key point 2"
    ]
}}

Return only JSON, no other content or markdown formatting. The number of key points should be determined by the content complexity and information density."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a professional text analysis expert, skilled at extracting and summarizing key information. Always base your key points on the provided content, do not add external information. Return only valid JSON without any markdown formatting."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=400
        )
        
        # Parse JSON response
        content = response.choices[0].message.content.strip()
        
        # Clean the response
        cleaned_content = clean_json_response(content)
        
        # Try to parse JSON
        try:
            result = json.loads(cleaned_content)
            key_points = result.get("key_points", [])
            
            # Filter out any non-string items and empty strings
            key_points = [point for point in key_points if isinstance(point, str) and point.strip()]
            
            return key_points
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            
            # Fallback: try to extract key points from lines
            lines = content.split('\n')
            key_points = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('{') and not line.startswith('}') and not line.startswith('```'):
                    # Clean possible numbering and symbols
                    cleaned = line.lstrip('0123456789.-*• ')
                    # Remove quotes and commas
                    cleaned = cleaned.strip('"",')
                    if cleaned and len(cleaned) > 10:  # Only keep substantial content
                        key_points.append(cleaned)
            return key_points
            
    except Exception as e:
        print(f"API call error: {e}")
        return []

def process_single_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single item to extract key points
    """
    try:
        # Get answer and question
        correct_answer = item['open_ended']['gpt40']['content']['correct_answer']
        question = item['open_ended']['gpt40']['content']['question']
        
        # Extract key points
        key_points = extract_keypoints_with_gpt4o(correct_answer, question)
        
        # Create a deep copy of the original item to preserve structure
        result_item = json.loads(json.dumps(item))
        
        # Add key_points to the content section
        result_item['open_ended']['gpt40']['content']['key_points'] = key_points
        
        return result_item
        
    except Exception as e:
        print(f"Error processing item: {e}")
        return None

async def process_golden_oe_data_async(input_file: str, output_file: str, sample_size: int = None, max_concurrent: int = 2000):
    """
    Process golden_oe.json file with async concurrent processing using ThreadPoolExecutor
    """
    
    print(f"Reading data file: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_questions = len(data)
    if sample_size:
        data = data[:sample_size]
        print(f"Processing first {sample_size} questions (total: {total_questions})")
    else:
        print(f"Processing all {total_questions} questions")
    
    print(f"Using async processing with max {max_concurrent} concurrent API calls")
    
    results = []
    processed_count = 0
    
    # Use ThreadPoolExecutor for concurrent API calls
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        # Submit all tasks
        loop = asyncio.get_event_loop()
        tasks = [loop.run_in_executor(executor, process_single_item, item) for item in data]
        
        # Process completed tasks
        for task in asyncio.as_completed(tasks):
            result = await task
            if result:
                results.append(result)
                processed_count += 1
                
                if processed_count % 20 == 0:
                    print(f"Processed {processed_count}/{len(data)} questions...")
                    # Save intermediate results
                    with open(f"{output_file}.temp", 'w', encoding='utf-8') as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Save final results
    print(f"Saving final results to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("Processing completed!")
    
    # Statistics
    total_keypoints = sum(len(item['open_ended']['gpt40']['content']['key_points']) for item in results)
    avg_keypoints = total_keypoints / len(results)
    print(f"Average key points per question: {avg_keypoints:.1f}")

def analyze_keypoints_distribution(output_file: str):
    """
    Analyze key points distribution
    """
    with open(output_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    keypoint_counts = [len(item['open_ended']['gpt40']['content']['key_points']) for item in data]
    
    print(f"\nKey Points Distribution Analysis:")
    print(f"Total questions: {len(data)}")
    print(f"Average key points: {sum(keypoint_counts)/len(keypoint_counts):.1f}")
    print(f"Minimum key points: {min(keypoint_counts)}")
    print(f"Maximum key points: {max(keypoint_counts)}")
    
    # Count distribution of different key point numbers
    from collections import Counter
    count_distribution = Counter(keypoint_counts)
    print(f"\nKey points count distribution:")
    for count in sorted(count_distribution.keys()):
        print(f"  {count} key points: {count_distribution[count]} questions")

def analyze_content_coverage(output_file: str, sample_size: int = 10):
    """
    Analyze how well key points cover the original answer content
    """
    with open(output_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    sample_data = data[:sample_size]
    total_coverage = 0
    
    print(f"\nContent Coverage Analysis (sample: {len(sample_data)} questions):")
    
    for i, item in enumerate(sample_data):
        answer = item['open_ended']['gpt40']['content']['correct_answer'].lower()
        key_points_text = ' '.join(item['open_ended']['gpt40']['content']['key_points']).lower()
        
        # Simple word-based coverage analysis
        answer_words = set(answer.split())
        keypoint_words = set(key_points_text.split())
        
        # Calculate coverage
        common_words = answer_words.intersection(keypoint_words)
        coverage = len(common_words) / len(answer_words) * 100 if answer_words else 0
        total_coverage += coverage
        
        print(f"Question {i+1}: {coverage:.1f}% coverage ({len(common_words)}/{len(answer_words)} words)")
    
    avg_coverage = total_coverage / len(sample_data)
    print(f"\nAverage coverage: {avg_coverage:.1f}%")

if __name__ == "__main__":
    # Check API key
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: Please set OPENAI_API_KEY in DATA/.env file")
        exit(1)
    
    input_file = "DATA/final_OE/golden_oe.json"
    output_file = "DATA/final_OE/golden_oe_with_keypoints.json"
    
    # Process all data with async processing
    asyncio.run(process_golden_oe_data_async(input_file, output_file, max_concurrent=2000))
    
    # Analyze results
    analyze_keypoints_distribution(output_file)
    analyze_content_coverage(output_file) 