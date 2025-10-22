#!/usr/bin/env python3
"""
Test script for key points extraction functionality
"""

import json
import os
from dotenv import load_dotenv
from extract_keypoints import extract_keypoints_with_gpt4o, analyze_content_coverage

# Load environment variables from DATA/.env file
load_dotenv('DATA/.env')

def test_keypoints_extraction():
    """Test key points extraction functionality"""
    
    # Check API key
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: Please set OPENAI_API_KEY in DATA/.env file")
        return
    
    # Read a sample for testing
    with open('DATA/final_OE/golden_oe.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Select first 3 questions for testing
    test_cases = data[:3]
    
    for i, item in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"Test Case {i+1}")
        print(f"{'='*60}")
        
        question = item['open_ended']['gpt40']['content']['question']
        answer = item['open_ended']['gpt40']['content']['correct_answer']
        
        print(f"Question: {question}")
        print(f"\nAnswer: {answer}")
        
        print(f"\nExtracting key points...")
        key_points = extract_keypoints_with_gpt4o(answer, question)
        
        print(f"\nExtracted key points ({len(key_points)}):")
        for j, point in enumerate(key_points, 1):
            print(f"  {j}. {point}")
        
        print(f"\nKey points analysis:")
        print(f"  - Count: {len(key_points)}")
        if key_points:
            print(f"  - Average length: {sum(len(p) for p in key_points)/len(key_points):.1f} characters")
        else:
            print(f"  - Average length: 0 characters (no key points extracted)")
        
        # Check content coverage
        answer_words = set(answer.lower().split())
        covered_words = set()
        for point in key_points:
            covered_words.update(point.lower().split())
        
        coverage = len(covered_words.intersection(answer_words)) / len(answer_words) * 100 if answer_words else 0
        print(f"  - Content coverage: {coverage:.1f}%")
        
        # Check for content-based extraction
        if key_points:
            print(f"  - Content-based extraction check:")
            for j, point in enumerate(key_points, 1):
                # Simple check: count how many words from the answer appear in this key point
                point_words = set(point.lower().split())
                common_words = answer_words.intersection(point_words)
                coverage_per_point = len(common_words) / len(point_words) * 100 if point_words else 0
                print(f"    Point {j}: {coverage_per_point:.1f}% content-based")
        else:
            print(f"  - Content-based extraction check: No key points to analyze")

def test_content_based_extraction():
    """Test if key points are truly based on answer content"""
    
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: Please set OPENAI_API_KEY in DATA/.env file")
        return
    
    # Read sample data
    with open('DATA/final_OE/golden_oe.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Test with one complex answer
    test_item = data[0]
    question = test_item['open_ended']['gpt40']['content']['question']
    answer = test_item['open_ended']['gpt40']['content']['correct_answer']
    
    print(f"\n{'='*60}")
    print("CONTENT-BASED EXTRACTION TEST")
    print(f"{'='*60}")
    
    print(f"Original answer length: {len(answer)} characters")
    print(f"Answer preview: {answer[:200]}...")
    
    key_points = extract_keypoints_with_gpt4o(answer, question)
    
    if key_points:
        print(f"\nExtracted {len(key_points)} key points:")
        for i, point in enumerate(key_points, 1):
            print(f"\n{i}. {point}")
            
            # Analyze content similarity
            answer_words = set(answer.lower().split())
            point_words = set(point.lower().split())
            common_words = answer_words.intersection(point_words)
            
            print(f"   - Words from answer: {len(common_words)}/{len(point_words)}")
            print(f"   - Content similarity: {len(common_words)/len(point_words)*100:.1f}%")
            
            if common_words:
                print(f"   - Key words: {', '.join(list(common_words)[:5])}{'...' if len(common_words) > 5 else ''}")
    else:
        print(f"\nNo key points extracted - API call may have failed")

if __name__ == "__main__":
    print("Starting key points extraction test...")
    test_keypoints_extraction()
    
    print(f"\n{'='*60}")
    print("Testing content-based extraction...")
    test_content_based_extraction() 