#!/usr/bin/env python3
"""
Simplified script to run key points extraction with async API concurrent processing
"""

import os
import sys
import asyncio
from dotenv import load_dotenv
from extract_keypoints import process_golden_oe_data_async, analyze_keypoints_distribution, analyze_content_coverage

# Load environment variables from DATA/.env file
load_dotenv('DATA/.env')

def main():
    # Check API key
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: Please set OPENAI_API_KEY in DATA/.env file")
        print("Example: Create DATA/.env file with OPENAI_API_KEY='your-api-key-here'")
        sys.exit(1)
    
    input_file = "DATA/final_OE/golden_oe.json"
    output_file = "DATA/final_OE/golden_oe_with_keypoints.json"  # Create a copy with key_points
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)
    
    print("Key Points Extraction Script")
    print("=" * 50)
    print("Using async API concurrent processing")
    print("Max concurrent API calls: 50")
    print("Processing all questions from golden_oe.json")
    
    # Ask for sample size
    sample_choice = input("\nProcess mode:\n1. Test (first 10 questions)\n2. Full processing (all questions)\nEnter choice (1/2): ").strip()
    
    if sample_choice == '1':
        sample_size = 10
        print("Processing first 10 questions for testing...")
    else:
        sample_size = None
        print("Processing all questions...")
    
    # Confirm start
    confirm = input("\nConfirm to start processing? (y/N): ").strip().lower()
    if confirm != 'y':
        print("Cancelled")
        sys.exit(0)
    
    try:
        # Process data with async API concurrent processing
        asyncio.run(process_golden_oe_data_async(input_file, output_file, sample_size, max_concurrent=2000))
        
        # Analyze results
        if os.path.exists(output_file):
            print("\n" + "="*50)
            print("ANALYSIS RESULTS")
            print("="*50)
            analyze_keypoints_distribution(output_file)
            analyze_content_coverage(output_file)
        
        print(f"\nProcessing completed! Results saved to: {output_file}")
        
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user")
        if os.path.exists(f"{output_file}.temp"):
            print(f"Intermediate results saved to: {output_file}.temp")
    except Exception as e:
        print(f"\nError during processing: {e}")
        if os.path.exists(f"{output_file}.temp"):
            print(f"Intermediate results saved to: {output_file}.temp")

if __name__ == "__main__":
    main() 