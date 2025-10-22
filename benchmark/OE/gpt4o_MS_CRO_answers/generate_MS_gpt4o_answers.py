#!/usr/bin/env python3
"""
Generate Golden Answers for OE Evaluation

This script generates high-quality golden answers from MS_MACRO dataset
for use in open-ended question evaluation. The focus is on generating
good answers, not evaluating them during generation.
"""

import json
import os
import random
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
import time

class GoldenAnswerGenerator:
    def __init__(self, ms_macro_file: str, output_dir: str, sample_size: int = 500):
        """
        Initialize the golden answer generator
        
        Args:
            ms_macro_file: Path to merged_MS_MACRO_cleaned.json
            output_dir: Output directory for generated golden answers
            sample_size: Number of QA pairs to process
        """
        self.ms_macro_file = ms_macro_file
        self.output_dir = output_dir
        self.sample_size = sample_size
        
        env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
        load_dotenv(dotenv_path=env_path)
        
        # use synchronous client
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Statistics
        self.stats = {
            "total_pairs": 0,
            "successful_generations": 0,
            "failed_generations": 0
        }

    def load_ms_macro_data(self) -> List[Dict]:
        """Load and sample MS_MACRO data"""
        print(f"Loading MS_MACRO data from {self.ms_macro_file}...")
        
        with open(self.ms_macro_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Total QA pairs in MS_MACRO: {len(data)}")
        
        # Sample data
        if len(data) > self.sample_size:
            sampled_data = random.sample(data, self.sample_size)
            print(f"Sampled {self.sample_size} QA pairs")
        else:
            sampled_data = data
            print(f"Using all {len(data)} QA pairs")
        
        return sampled_data

    def generate_golden_answer(self, question: str, context: str) -> str:
        """
        Generate a high-quality answer using GPT-4o based on context only
        
        Args:
            question: The question to answer
            context: The context/passage to use
        """
        try:
            prompt = f"""Based on the given context, provide a comprehensive answer to the question.

Question: {question}

Context: {context}

Please provide a detailed answer that:
1. Directly addresses the question
2. Uses information from the context
3. Is comprehensive and well-structured
4. Provides specific details and examples when available
5. Is suitable as a high-quality reference answer

Answer:"""

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert at providing accurate, comprehensive answers based on given context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400,
                temperature=0.1  # Low temperature for consistency
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"Error generating answer: {e}")
            return ""

    def generate_golden_answers_dataset(self) -> List[Dict]:
        """Generate golden answers for the entire dataset"""
        print("Starting Golden Answer Generation...")
        
        # Load data
        data = self.load_ms_macro_data()
        self.stats["total_pairs"] = len(data)
        
        golden_answers_dataset = []
        
        for i, qa_pair in enumerate(tqdm(data, desc="Generating golden answers")):
            question = qa_pair.get("anchor", "")
            context = qa_pair.get("positive", "")
            
            if not question or not context:
                continue
            
            # Generate golden answer
            golden_answer = self.generate_golden_answer(question, context)
            
            if golden_answer:
                self.stats["successful_generations"] += 1
                
                # Create dataset entry
                entry = {
                    "id": f"gpt4o_{i:04d}",
                    "question": question,
                    "context": context,
                    "gpt4o_answer": golden_answer,
                    "human_answer": qa_pair.get("answers", [""])[0] if qa_pair.get("answers") else "",
                    "metadata": {
                        "source": "ms_macro",
                        "generated_by": "gpt-4o"
                    }
                }
                
                golden_answers_dataset.append(entry)
            else:
                self.stats["failed_generations"] += 1
            
            # Add delay to avoid rate limiting
            time.sleep(0.1)
        
        return golden_answers_dataset

    def save_golden_answers(self, dataset: List[Dict]):
        """Save the generated GPT4o answers dataset"""
        # Save full dataset
        dataset_file = os.path.join(self.output_dir, "gpt4o_answers_dataset.json")
        with open(dataset_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        # Save statistics
        stats_file = os.path.join(self.output_dir, "generation_stats.json")
        stats = {
            "generation_stats": self.stats,
            "success_rate": self.stats["successful_generations"] / self.stats["total_pairs"] if self.stats["total_pairs"] > 0 else 0,
            "dataset_info": {
                "total_entries": len(dataset),
                "output_file": dataset_file
            }
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"GPT4o answers dataset saved to: {dataset_file}")
        print(f"Generation statistics saved to: {stats_file}")

    def print_summary(self):
        """Print generation summary"""
        print("\n" + "="*60)
        print("GPT4o Answer Generation Summary")
        print("="*60)
        print(f"Total QA pairs processed: {self.stats['total_pairs']}")
        print(f"Successful generations: {self.stats['successful_generations']}")
        print(f"Failed generations: {self.stats['failed_generations']}")
        print(f"Success rate: {self.stats['successful_generations'] / self.stats['total_pairs']:.2%}")
        
        print(f"\nNext Steps:")
        print(f"1. Compare GPT4o answers with human answers")
        print(f"2. Apply standard evaluation metrics (BLEU, ROUGE, BERTScore, etc.)")
        print(f"3. Assess if GPT4o can replace human answers for OE evaluation")

def main():
    """Main function"""
    # Configuration
    ms_macro_file = "merged_MS_MACRO_cleaned.json"
    output_dir = "gpt4o_answers"
    sample_size = 500
    
    # Check if MS_MACRO file exists
    if not os.path.exists(ms_macro_file):
        print(f"Error: {ms_macro_file} not found in current directory")
        return
    
    # Initialize generator
    generator = GoldenAnswerGenerator(ms_macro_file, output_dir, sample_size)
    
    # Generate golden answers
    dataset = generator.generate_golden_answers_dataset()
    
    # Save results
    generator.save_golden_answers(dataset)
    
    # Print summary
    generator.print_summary()

if __name__ == "__main__":
    main() 