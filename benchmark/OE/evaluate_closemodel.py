import json
import os
import argparse
from typing import List, Dict, Any
import openai
from tqdm import tqdm
# import anthropic
# import google.generativeai as genai
import requests
import time
import re
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import random
import asyncio

class OEGeneratorBenchmark:
    # load .env file
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    load_dotenv(dotenv_path=env_path)
    
    def __init__(self, test_set_path: str):
        """
        Initialize the OE generator benchmark
        
        Args:
            test_set_path: Path to the test set (base/golden/mix)
        """
        self.test_set = self._load_test_set(test_set_path)
        self.setting = os.path.basename(test_set_path).split('_')[2]  # base/golden/mix
        
        # API keys
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        # self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        # self.google_api_key = os.getenv("GOOGLE_API_KEY")
        
        # Thread safety
        self.print_lock = Lock()
        self.results_lock = Lock()
        
        # Set random seed for reproducibility
        # random.seed(42)
    
    def _load_test_set(self, test_set_path: str) -> List[Dict]:
        """Load the test set"""
        with open(test_set_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    async def generate_answer(self, question: str, context: List[str], model: str) -> str:
        """Generate answer for open-ended question"""
        # Build context string
        context_str = ""
        if context:  # For mix and golden settings
            context_str = "\n".join(context)
        
        # Build prompt based on setting
        if self.setting == "base":
            prompt = f"""
            Answer the following open-ended question based on your knowledge.
            Provide a comprehensive, well-structured response.

            Question: {question}

            Please provide your answer in the following format:
            Answer: [Your comprehensive response]
            Confidence: [High/Medium/Low]
            Explanation: [Your reasoning for this answer]

            Your answer should:
            1. Be comprehensive and well-structured
            2. Include relevant details and examples
            3. Demonstrate understanding of the topic
            4. Be based on your knowledge and understanding
            """
        else:  # mix and golden settings
            prompt = f"""
            You are given a reference passage and an open-ended question.
            While you should primarily use the information from the passage to answer the question,
            you can also use your knowledge to help understand and interpret the passage.

            Reference Passage:
            {context_str}

            Question: {question}

            Please provide your answer in the following format:
            Answer: [Your comprehensive response]
            Confidence: [High/Medium/Low]
            Explanation: [Your reasoning for this answer]

            Your answer should:
            1. First, use information from the passage to address the question
            2. Then, use your knowledge to help interpret and expand on the passage information
            3. Be comprehensive and well-structured
            4. Include specific evidence and examples when relevant

            For confidence levels:
            - High: The answer is clearly supported by the passage and your knowledge
            - Medium: The answer is somewhat supported, but there might be some ambiguity
            - Low: The answer is not clearly supported, and you're making an educated guess
            """
        
        # Call the appropriate model
        return await self._call_model(model, prompt)
    
    async def _call_model(self, model_name: str, prompt: str) -> str:
        """Call the appropriate model based on model name.
        
        Args:
            model_name: Name of the model to use
            prompt: The prompt to send to the model
            
        Returns:
            str: Model's response
        """
        if model_name in ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"]:  # Add gpt-4o to supported models
            return await self._call_openai(model_name, prompt)
        elif model_name in ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"]:  # Add Gemini models
            return await self._call_google(model_name, prompt)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    async def _call_openai(self, model: str, prompt: str) -> str:
        """Call OpenAI API asynchronously."""
        client = openai.AsyncOpenAI(api_key=self.openai_api_key)
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers open-ended questions based on the given context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,  # Higher temperature for more creative OE answers
                max_tokens=2000   # Increase max_tokens for open-ended answers
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling OpenAI API for model {model}: {str(e)}")
            raise
    
    # def _call_anthropic(self, prompt: str) -> str:
    #     client = anthropic.Anthropic(api_key=self.anthropic_api_key)
    #     response = client.messages.create(
    #         model="claude-2",
    #         messages=[{"role": "user", "content": prompt}]
    #     )
    #     return response.content[0].text
    
    # async def _call_google(self, model_name: str, prompt: str) -> str:
    #     """Call Google Gemini API asynchronously with retry logic."""
    #     max_retries = 3
    #     retry_delay = 1  # Start with 1 second delay
        
    #     for attempt in range(max_retries):
    #         try:
    #             genai.configure(api_key=self.google_api_key)
    #             model = genai.GenerativeModel(model_name)
                
    #             # Configure generation parameters for more deterministic responses
    #             generation_config = genai.types.GenerationConfig(
    #                 temperature=0.7,  # Higher temperature for more creative OE answers
    #                 max_output_tokens=2000,  # Increase max_tokens for open-ended answers
    #             )
                
    #             # Use async generate_content
    #             response = await model.generate_content_async(
    #                 prompt,
    #                 generation_config=generation_config
    #             )
    #             return response.text
                
    #         except Exception as e:
    #             error_str = str(e).lower()
    #             # Check for rate limit or quota exceeded errors
    #             if ("quota" in error_str or "rate" in error_str or "limit" in error_str) and attempt < max_retries - 1:
    #                 wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
    #                 print(f"Rate limit hit for {model_name}, waiting {wait_time}s before retry...")
    #                 await asyncio.sleep(wait_time)
    #                 continue
    #             else:
    #                 print(f"Error calling Google Gemini API for model {model_name}: {str(e)}")
    #                 raise

#     def _generate_prompt(self, item: Dict[str, Any]) -> str:
#         """Generate prompt for the model based on test set item."""
#         # Get question from OE format
#         if "open_ended" in item and "gpt40" in item["open_ended"]:
#             question = item["open_ended"]["gpt40"]["content"]["question"]
#         else:
#             raise ValueError("Invalid test set format: missing open_ended or gpt40 content")

#         # Get retrieved documents if available
#         retrieved_docs = item.get("retrieved_documents", [])
        
#         # Build prompt
#         prompt = f"""Please answer the following open-ended question. Provide a comprehensive, well-structured response.

# Question: {question}

# """
#         if retrieved_docs:
#             prompt += "\nRelevant information:\n"
#             for i, doc in enumerate(retrieved_docs, 1):
#                 prompt += f"{i}. {doc}\n"
        
#         prompt += "\nPlease provide your answer in the following format:\nAnswer: [Your comprehensive response]\nExplanation: [Your reasoning]"
        
#         return prompt


    async def process_single_item(self, item: Dict[str, Any], model_name: str, item_index: int, total: int) -> Dict[str, Any]:
        """Process a single test item."""
        try:
            # Get question from OE format
            question = item["open_ended"]["gpt40"]["content"]["question"]
            original_query = item["original_query"]
            
            # Get context based on setting
            if self.setting == "base":
                context = []
            else:
                context = [item["passage"]] if "passage" in item else []
                if not context:
                    return None  # Skip items without passage for mix/golden settings
            
            # Generate answer
            raw_answer = await self.generate_answer(question, context, model_name)
            
            # Parse the answer, confidence, and explanation
            answer, explanation = self._parse_response(raw_answer)
            if not answer:
                # If parsing fails, use the full response as answer
                answer = raw_answer.strip()
            
            # Create result
            result = {
                "original_query": original_query,
                "question": question,
                "model_answer": answer,
                "explanation": explanation or raw_answer,  # Use parsed explanation or full response
                "passage": context[0] if context else ''
            }
            
            return result
            
        except Exception as e:
            return None

    def _parse_response(self, response: str) -> tuple[str, str]:
        """Parse the model's response to extract answer and explanation.
        
        Args:
            response: The model's full response text
            
        Returns:
            tuple: (answer_text, explanation)
        """
        # Clean the response
        response = response.strip()
        
            # Try to find answer and explanation
        answer_match = re.search(r"Answer:\s*(.*?)(?=\nExplanation:|$)", response, re.DOTALL | re.IGNORECASE)
        explanation_match = re.search(r"Explanation:\s*(.*?)(?:\n|$)", response, re.DOTALL | re.IGNORECASE)
        
        answer = answer_match.group(1).strip() if answer_match else response
        explanation = explanation_match.group(1).strip() if explanation_match else ""
        
        return answer, explanation

    async def process_test_set(self, test_set: List[Dict[str, Any]], model_name: str, num_tests: int = None, concurrency: int = 500) -> List[Dict[str, Any]]:
        """Process test set with specified model using asyncio.
        
        Args:
            test_set: List of test cases
            model_name: Name of the model to use
            num_tests: Number of test cases to process. If None, process all.
            concurrency: Maximum number of concurrent requests
        """
        results = []
        total = len(test_set)
        
        # If num_tests is specified, randomly select that many cases
        if num_tests is not None:
            test_set = random.sample(test_set, min(num_tests, total))
            total = len(test_set)
            print(f"\nRandomly selected {total} items for testing...")
        else:
            print(f"\nProcessing all {total} items...")
        
        # Create semaphore to limit concurrency
        sem = asyncio.Semaphore(concurrency)
        
        async def process_with_semaphore(item, index):
            async with sem:
                return await self.process_single_item(item, model_name, index, total)
        
        # Create tasks
        tasks = [
            process_with_semaphore(item, i+1)
            for i, item in enumerate(test_set)
        ]
        
        # Process all tasks
        for result in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Processing {self.setting} test set"):
            result = await result
            if result is not None:
                results.append(result)
        
        # Calculate statistics
        total_answers = len(results)
        if total_answers == 0:
            print(f"\nNo valid results for {self.setting} setting")
            return results
            
        print(f"\nStatistics for {self.setting}:")
        print(f"Total processed: {total_answers}/{total} ({total_answers/total*100:.1f}%)")
        
        # Calculate average answer length
        avg_answer_length = sum(len(r["model_answer"]) for r in results) / total_answers if total_answers > 0 else 0
        print(f"Average answer length: {avg_answer_length:.1f} characters")
        
        return results

def load_test_set(file_path: str) -> List[Dict[str, Any]]:
    """Load test set from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_results(results: List[Dict[str, Any]], output_path: str):
    """Save results to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def print_test_set_statistics(test_sets: List[str]):
    """Print statistics for all test sets."""
    print("\nTest set statistics:")
    for test_set in test_sets:
        with open(test_set, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Get all questions
            questions = [item["open_ended"]["gpt40"]["content"]["question"] for item in data]
            # Count unique questions
            unique_questions = set(questions)
            print(f"\n{os.path.basename(test_set)}:")
            print(f"Total items: {len(data)}")
            print(f"Unique questions: {len(unique_questions)}")
            # Print some example questions
            print("Example questions:")
            for q in list(unique_questions)[:3]:
                print(f"- {q}")

def main():
    # Specify the model to use
    # Options: "gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o", "gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"
    model_name = "gpt-4o"  # Change this to the model you want to use
    
    # Set number of test cases to process (None for all)
    num_tests = None  # Process all test cases
    
    # Set concurrency limit (Gemini has rate limits, so use lower concurrency)
    concurrency = 100 if model_name.startswith("gemini") else 2000  # Lower for Gemini API
    
    # Set paths - Updated for OE test sets
    test_sets = [
        "/home/shared/RAG_DATA/benchmark/OE/generated_test_sets/test_set_base_simple.json",
        "/home/shared/RAG_DATA/benchmark/OE/generated_test_sets/test_set_golden_simple.json",
        "/home/shared/RAG_DATA/benchmark/OE/generated_test_sets/test_set_mix_simple.json"
    ]
    
    # Create output directory
    output_dir = f"/home/shared/RAG_DATA/benchmark/testOE_set/{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Print statistics for all test sets
    print_test_set_statistics(test_sets)
    
    print(f"\n{'='*50}")
    print(f"Starting OE benchmark with {num_tests if num_tests else 'all'} test cases")
    print(f"{'='*50}\n")
    
    # Run benchmarks
    async def run_benchmarks():
        for test_set in test_sets:
            setting = os.path.basename(test_set).split('_')[2]  # base/golden/mix
            print(f"\nTesting {model_name} on {setting} setting...")
            
            benchmark = OEGeneratorBenchmark(test_set)
            results = await benchmark.process_test_set(benchmark.test_set, model_name, num_tests, concurrency)
            
            # Save results for this setting
            output_path = os.path.join(output_dir, f"{setting}_results.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"\nResults for {setting} saved to: {output_path}")
    
    # Run the async main function
    asyncio.run(run_benchmarks())

if __name__ == "__main__":
    main()