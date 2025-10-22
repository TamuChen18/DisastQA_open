import json
import openai
import time
import asyncio
import argparse
from typing import List, Dict, Any
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure OpenAI API
# Get API key from environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.") 
def generate_oe_answer(question: str, passage: str, model: str = "gpt-4o") -> str:
    """
    Generate standard answers for OE questions using LLM based on the provided passage
    
    Args:
        question: OE question
        passage: Provided passage
        model: Model to use
    
    Returns:
        Generated answer
    """
    
    prompt = f"""
You are a QA assistant generating reference answers. 
Your task is to answer the question **using only the content of the passage**.

- Do not use any external knowledge.
- Reuse or rephrase exact content from the passage.
- If the passage does not contain enough information to answer the question, clearly say so (e.g., "The passage does not provide sufficient information to answer this question").

This answer will be used as a reference answer for later evaluation, so it must be grounded only in the passage.

Passage:
{passage}

Question:
{question}

Answer:"""



    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a professional Q&A assistant who answers questions strictly based on the provided passage."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Low temperature to ensure answer consistency
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating answer: {e}")
        return ""

async def generate_oe_answer_async(question: str, passage: str, model: str = "gpt-4o") -> str:
    """
    Async version of generate_oe_answer for high concurrency
    
    Args:
        question: OE question
        passage: Provided passage
        model: Model to use
    
    Returns:
        Generated answer
    """
    
    prompt = f"""
You are a QA assistant generating reference answers. 
Your task is to answer the question **using only the content of the passage**.

- Do not use any external knowledge.
- Reuse or rephrase exact content from the passage.
- If the passage does not contain enough information to answer the question, clearly say so (e.g., "The passage does not provide sufficient information to answer this question").

This answer will be used as a reference answer for later evaluation, so it must be grounded only in the passage.

Passage:
{passage}

Question:
{question}

Answer:"""


    try:
        # Use OpenAI's async client
        client = openai.AsyncOpenAI()
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a professional Q&A assistant who answers questions strictly based on the provided passage."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating answer: {e}")
        return ""

async def process_oe_data_high_concurrency(input_file: str, output_file: str, model: str = "gpt-4o", max_concurrent: int = 20):
    """
    High concurrency processing of OE data using async/await
    Only process samples that don't have llm_answer or have empty llm_answer
    
    Args:
        input_file: Input OE data file
        output_file: Output file with LLM answers in llm_answer field
        model: Model to use
        max_concurrent: Maximum concurrent API requests
    """
    
    # Read data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Filter samples that need processing (no llm_answer or empty llm_answer)
    samples_to_process = []
    for i, item in enumerate(data):
        llm_answer = item["open_ended"]["gpt40"]["content"].get("llm_answer", "")
        if not llm_answer or llm_answer == "":
            samples_to_process.append((i, item))
    
    print(f"Found {len(samples_to_process)} samples that need processing out of {len(data)} total samples")
    print(f"Starting high concurrency processing with {max_concurrent} concurrent requests...")
    
    if len(samples_to_process) == 0:
        print("No samples need processing. All samples already have llm_answer.")
        return data
    
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_single_item(index, item):
        async with semaphore:
            question = item["open_ended"]["gpt40"]["content"]["question"]
            passage = item["passage"]
            
            llm_answer = await generate_oe_answer_async(question, passage, model)
            return index, llm_answer
    
    # Create tasks only for samples that need processing
    tasks = [process_single_item(index, item) for index, item in samples_to_process]
    
    # Execute all tasks concurrently
    print("Executing tasks concurrently...")
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    completed = 0
    for i, result in enumerate(results):
        original_index, item = samples_to_process[i]
        if isinstance(result, Exception):
            print(f"  Error processing sample {original_index + 1}: {result}")
            # Set empty llm_answer
            content = data[original_index]["open_ended"]["gpt40"]["content"]
            content["llm_answer"] = ""
        else:
            index, llm_answer = result
            # Set LLM answer
            content = data[index]["open_ended"]["gpt40"]["content"]
            content["llm_answer"] = llm_answer
            completed += 1
            
            # Log progress
            if completed % 10 == 0:
                print(f"  Completed {completed}/{len(samples_to_process)} samples")
    
    # Save final results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"All {len(samples_to_process)} samples processed!")
    return data

async def test_oe_generation(input_file: str, output_file: str, test_size: int = 10, model: str = "gpt-4o", max_concurrent: int = 5):
    """
    Test OE answer generation with a small subset of data
    
    Args:
        input_file: Input OE data file
        output_file: Output file with test results
        test_size: Number of samples to test
        model: Model to use
        max_concurrent: Maximum concurrent API requests for testing
    """
    
    # Read data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Take only first test_size samples
    test_data = data[:test_size]
    
    print(f"Testing with {len(test_data)} samples using {max_concurrent} concurrent requests...")
    print(f"Model: {model}")
    
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_single_item(item, index):
        async with semaphore:
            question = item["open_ended"]["gpt40"]["content"]["question"]
            passage = item["passage"]
            
            print(f"  Processing sample {index + 1}/{len(test_data)}...")
            llm_answer = await generate_oe_answer_async(question, passage, model)
            return index, llm_answer
    
    # Create test tasks
    tasks = [process_single_item(item, i) for i, item in enumerate(test_data)]
    
    # Execute test tasks
    print("Executing test tasks...")
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    completed = 0
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"  ❌ Error processing sample {i + 1}: {result}")
            # Rename answer to llm_answer and set empty
            content = test_data[i]["open_ended"]["gpt40"]["content"]
            if "answer" in content:
                content["llm_answer"] = ""
                del content["answer"]
        else:
            index, llm_answer = result
            # Rename answer to llm_answer and set LLM answer
            content = test_data[index]["open_ended"]["gpt40"]["content"]
            if "answer" in content:
                content["llm_answer"] = llm_answer
                del content["answer"]
            completed += 1
            print(f"  ✅ Sample {index + 1} completed")
    
    # Save test results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nTest completed!")
    print(f"Successfully processed: {completed}/{len(test_data)} samples")
    print(f"Results saved to: {output_file}")
    
    # Show sample results
    print(f"\nSample results:")
    for i, item in enumerate(test_data[:3]):  # Show first 3 results
        print(f"\n--- Sample {i + 1} ---")
        print(f"Question: {item['open_ended']['gpt40']['content']['question'][:100]}...")
        print(f"Answer: {item['open_ended']['gpt40']['content']['llm_answer'][:200]}...")
    
    return test_data

if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='Generate OE answers with high concurrency')
    parser.add_argument('--test', action='store_true', help='Run in test mode with small sample size')
    parser.add_argument('--test-size', type=int, default=10, help='Number of samples for test mode (default: 10)')
    parser.add_argument('--input', type=str, default='sampled_oe_independent.json', help='Input file path')
    parser.add_argument('--output', type=str, default='ground_truth_check_answers.json', help='Output file path')
    parser.add_argument('--model', type=str, default='gpt-4o', help='Model to use')
    parser.add_argument('--max-concurrent', type=int, default=1000, help='Maximum concurrent requests')
    parser.add_argument('--test-concurrent', type=int, default=5, help='Maximum concurrent requests for test mode')
    
    args = parser.parse_args()
    
    if args.test:
        # Test mode
        print("Running in TEST mode...")
        asyncio.run(test_oe_generation(
            input_file=args.input,
            output_file=f"test_{args.output}",
            test_size=args.test_size,
            model=args.model,
            max_concurrent=args.test_concurrent
        ))
    else:
        # Full processing mode
        print("Running in FULL processing mode...")
        asyncio.run(process_oe_data_high_concurrency(
            input_file=args.input,
            output_file=args.output,
            model=args.model,
            max_concurrent=args.max_concurrent
        )) 