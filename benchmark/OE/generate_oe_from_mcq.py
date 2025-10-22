import json
import os
from typing import List, Dict, Optional
from openai import OpenAI
from tqdm import tqdm
import time
import random
from dotenv import load_dotenv
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_json_file(file_path: str) -> List[Dict]:
    """Load a JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON file {file_path}: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise

def save_json_file(file_path: str, data: List[Dict]):
    """Save data to a JSON file"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving JSON file {file_path}: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise

class OEFromMCQGenerator:
    def __init__(self, mcq_dir: str, output_dir: str):
        """Initialize OE generator from MCQ test sets"""
        self.mcq_dir = mcq_dir
        self.output_dir = output_dir
        
        # load .env file from benchmark directory (parent directory)
        env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
        load_dotenv(dotenv_path=env_path)
        
        # use synchronous client
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Create semaphore for API rate limiting
        self.api_semaphore = threading.Semaphore(2000)
        
        # Statistics
        self.stats = {
            "total_mcq_cases": 0,
            "successful_oe_generations": 0,
            "failed_generations": 0,
            "skipped_cases": 0,
            "quality_scores": {
                "clarity": [],
                "difficulty": [],
                "relevance": [],
                "educational_value": [],
                "answer_quality": [],
                "cognitive_level": [],
                "answerability": [],
                "final_scores": []
            }
        }

    def load_mcq_test_sets(self) -> Dict[str, List[Dict]]:
        """Load MCQ test sets for all settings"""
        test_sets = {}
        
        for setting in ["base", "golden", "mix"]:
            file_path = os.path.join(self.mcq_dir, f"test_set_{setting}_simple.json")
            if os.path.exists(file_path):
                test_sets[setting] = load_json_file(file_path)
                print(f"Loaded {len(test_sets[setting])} {setting} cases from MCQ")
            else:
                print(f"Warning: {file_path} not found")
                test_sets[setting] = []
        
        return test_sets

    def generate_open_ended_question(self, user_query: str, passage: Optional[str], model: str = "gpt-4o-mini") -> Optional[Dict]:
        """Generate open-ended question and answer based on user_query and passage"""
        
        if passage:
            # With passage (golden/mix cases)
            prompt = f"""
            You are a helpful assistant that creates high-quality open-ended questions and comprehensive answers for evaluation and educational benchmarking.

            Given the following **user query** and **reference document**, create an improved open-ended question and provide a comprehensive answer:

            ---
            User Query:
            \"\"\"{user_query}\"\"\"

            Reference Document:
            \"\"\"{passage}\"\"\"
            ---

            Instructions:

            1. **Question Generation**: 
               - Refine the user query into a clear, comprehensive open-ended question
               - The question should encourage detailed, analytical responses
               - Make it specific enough to be answerable but broad enough to allow comprehensive discussion
               - The question should require synthesis of information from the reference document

            2. **Answer Generation**:
               - Provide a comprehensive, well-structured answer based on the reference document
               - The answer should be detailed (3-5 paragraphs)
               - Include specific evidence and examples from the reference document
               - Organize the response logically with clear reasoning
               - Address multiple aspects of the question when relevant

            3. **Quality Requirements**:
               - Do **not** copy the reference document verbatim
               - Use your own words while maintaining accuracy
               - Ensure the answer directly addresses the refined question
               - Include relevant context and explanations

            Return only the following JSON format:

            {{
            "question": "...",
            "answer": "...",
            "key_points": ["point1", "point2", "point3"],
            "reasoning": "Brief explanation of how the answer addresses the question"
            }}
            """
        else:
            # Without passage (base cases)
            prompt = f"""
            You are a helpful assistant that creates high-quality open-ended questions and comprehensive answers for evaluation and educational benchmarking.

            Given the following **user query**, create an improved open-ended question and provide a comprehensive answer based on your general knowledge:

            ---
            User Query:
            \"\"\"{user_query}\"\"\"
            ---

         Instructions:

        1. **Question Generation**: 
           - Refine the user query into a clear, comprehensive open-ended question
           - The question should encourage detailed, analytical responses
           - Make it specific enough to be answerable but broad enough to allow comprehensive discussion
           - The question should require synthesis of information from the reference document

        2. **Answer Generation**:
           - Provide a comprehensive, well-structured answer based on the reference document
           - The answer should be detailed (3-5 paragraphs)
           - Include specific evidence and examples from the reference document
           - Organize the response logically with clear reasoning
           - Address multiple aspects of the question when relevant

        3. **Quality Requirements**:
           - Do **not** copy the reference document verbatim
           - Use your own words while maintaining accuracy
           - Ensure the answer directly addresses the refined question
           - Include relevant context and explanations

        Return only the following JSON format:

        {{
        "question": "...",
        "answer": "...",
        "key_points": ["point1", "point2", "point3"],
        "reasoning": "Brief explanation of how the answer addresses the question"
        }}
        """
        
        try:
            with self.api_semaphore:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that generates high-quality open-ended questions and comprehensive answers. Always respond with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={ "type": "json_object" }
                )
            
            # Try to parse the response
            try:
                question_data = json.loads(response.choices[0].message.content)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON response: {str(e)}")
                print(f"Raw response: {response.choices[0].message.content}")
                return None
            
            # Validate the question data structure
            required_keys = ["question", "answer", "key_points", "reasoning"]
            if not all(key in question_data for key in required_keys):
                print(f"Missing required keys in question data: {question_data}")
                return None
            
            if not isinstance(question_data["key_points"], list):
                print(f"key_points should be a list: {question_data['key_points']}")
                return None
            
            return question_data
                
        except Exception as e:
            print(f"Error during question generation: {str(e)}")
            return None

    def generate_oe_question_only(self, user_query: str, passage: Optional[str], model: str = "gpt-4o-mini") -> Optional[Dict]:
        """Generate open-ended question wording only (without answer)"""
        prompt = f"""
        You are a helpful assistant that creates high-quality open-ended questions and comprehensive answers for evaluation and educational benchmarking.

        Given the following **user query** and **reference document**, create an improved open-ended question wording:

        ---
        User Query:
        \"\"\"{user_query}\"\"\"

        Reference Document:
        \"\"\"{passage}\"\"\"
        ---

        Instructions:

        1. **Question Generation**: 
           - Refine the user query into a clear, comprehensive open-ended question wording
           - The wording should encourage detailed, analytical responses
           - Make it specific enough to be answerable but broad enough to allow comprehensive discussion
           - The wording should require synthesis of information from the reference document

        2. **Answer Generation**:
           - Provide a comprehensive, well-structured answer based on the reference document
           - The answer should be detailed (3-5 paragraphs)
           - Include specific evidence and examples from the reference document
           - Organize the response logically with clear reasoning
           - Address multiple aspects of the question when relevant

        3. **Quality Requirements**:
           - Do **not** copy the reference document verbatim
           - Use your own words while maintaining accuracy
           - Ensure the answer directly addresses the refined question
           - Include relevant context and explanations

        Return only the following JSON format:

        {{
        "question": "...",
        "reasoning": "Brief explanation of how the question wording addresses the question"
        }}
        """
        
        try:
            with self.api_semaphore:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that generates high-quality open-ended questions and comprehensive answers. Always respond with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={ "type": "json_object" }
                )
            
            # Try to parse the response
            try:
                question_data = json.loads(response.choices[0].message.content)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON response: {str(e)}")
                print(f"Raw response: {response.choices[0].message.content}")
                return None
            
            # Validate the question data structure
            required_keys = ["question", "reasoning"]
            if not all(key in question_data for key in required_keys):
                print(f"Missing required keys in question data: {question_data}")
                return None
            
            return question_data
                
        except Exception as e:
            print(f"Error during question wording generation: {str(e)}")
            return None

    def generate_oe_answer_only(self, oe_question: Dict, passage: Optional[str], model: str = "gpt-4o-mini") -> Optional[Dict]:
        """Generate open-ended answer only (using a pre-generated question wording)"""
        prompt = f"""
        You are a helpful assistant that creates high-quality open-ended questions and comprehensive answers for evaluation and educational benchmarking.

        Given the following **open-ended question wording** and **reference document**, provide a comprehensive answer:

        ---
        Open-Ended Question Wording:
        \"\"\"{oe_question["question"]}\"\"\"

        Reference Document:
        \"\"\"{passage}\"\"\"
        ---

        Instructions:

        1. **Answer Generation**:
           - Provide a comprehensive, well-structured answer based on the reference document
           - The answer should be detailed (3-5 paragraphs)
           - Include specific evidence and examples from the reference document
           - Organize the response logically with clear reasoning
           - Address multiple aspects of the question when relevant

        2. **Quality Requirements**:
           - Do **not** copy the reference document verbatim
           - Use your own words while maintaining accuracy
           - Ensure the answer directly addresses the refined question
           - Include relevant context and explanations

        Return only the following JSON format:

        {{
        "answer": "...",
        "key_points": ["point1", "point2", "point3"],
        "reasoning": "Brief explanation of how the answer addresses the question"
        }}
        """
        
        try:
            with self.api_semaphore:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that generates high-quality open-ended questions and comprehensive answers. Always respond with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={ "type": "json_object" }
                )
            
            # Try to parse the response
            try:
                answer_data = json.loads(response.choices[0].message.content)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON response: {str(e)}")
                print(f"Raw response: {response.choices[0].message.content}")
                return None
            
            # Validate the answer data structure
            required_keys = ["answer", "key_points", "reasoning"]
            if not all(key in answer_data for key in required_keys):
                print(f"Missing required keys in answer data: {answer_data}")
                return None
            
            if not isinstance(answer_data["key_points"], list):
                print(f"key_points should be a list: {answer_data['key_points']}")
                return None
            
            return answer_data
                
        except Exception as e:
            print(f"Error during answer generation: {str(e)}")
            return None

    def process_mcq_case(self, mcq_case: Dict, setting: str) -> Optional[Dict]:
        """Process a single MCQ case to generate corresponding OE case with quality assessment and retry"""
        try:
            original_query = mcq_case["original_query"]
            passage = mcq_case.get("passage", None)  # base cases don't have passage
            
            # Try to generate a high quality question with retries
            max_retries = 3
            best_oe_content = None
            best_quality = None
            best_quality_score = 0
            
            for retry in range(max_retries):
                print(f"\nAttempt {retry + 1}/{max_retries} for query: {original_query[:50]}...")
                
                # Generate OE question and answer
                oe_content = self.generate_with_retry(self.generate_open_ended_question, original_query, passage)
                
                if oe_content is None:
                    print("Failed to generate OE question")
                    if retry < max_retries - 1:
                        print("Retrying OE generation...")
                        continue
                    else:
                        print("Max retries reached, skipping this case")
                        self.stats["failed_generations"] += 1
                        return None
                
                # Assess the quality of generated OE
                oe_quality = self.assess_question_quality(
                    oe_content["question"],
                    oe_content["answer"],
                    "gpt-4o-mini"
                )
                
                print(f"OE quality score: {oe_quality['final_score']}, is_valid: {oe_quality['is_valid']}")
                
                # Update statistics
                for metric in ["clarity", "difficulty", "relevance", "educational_value", "answer_quality", "cognitive_level", "answerability"]:
                    if metric in oe_quality:
                        self.stats["quality_scores"][metric].append(oe_quality[metric])
                self.stats["quality_scores"]["final_scores"].append(oe_quality["final_score"])
                
                # Keep track of the best quality result
                if oe_quality["final_score"] > best_quality_score:
                    best_quality_score = oe_quality["final_score"]
                    best_oe_content = oe_content
                    best_quality = oe_quality
                
                # If quality is good enough, use it immediately
                if oe_quality["is_valid"]:
                    print("Quality meets threshold, using this result")
                    best_oe_content = oe_content
                    best_quality = oe_quality
                    break
                
                # If not the last retry, continue
                if retry < max_retries - 1:
                    print("Quality below threshold, retrying...")
                    continue
            
            # Use the best result (either valid or highest score)
            if best_oe_content is None:
                print("No valid OE content generated")
                self.stats["failed_generations"] += 1
                return None
            
            # Create OE test case with detailed metadata
            oe_case = {
                "original_query": original_query,
                "setting": setting,
                "metadata": {
                    "source": "mcq_generated",
                    "setting": setting,
                    "has_passage": passage is not None,
                    "quality_score": best_quality["final_score"],
                    "generated_quality": best_quality,
                    "difficulty_level": "easy" if best_quality["difficulty"] <= 3 else "hard",
                    "is_valid": best_quality["is_valid"]
                },
                "open_ended": {
                    "gpt40": {
                        "content": best_oe_content
                    }
                }
            }
            
            # Add passage for golden and mix cases
            if setting in ["golden", "mix"] and passage:
                oe_case["passage"] = passage
            
            self.stats["successful_oe_generations"] += 1
            return oe_case
            
        except Exception as e:
            print(f"Error processing MCQ case: {str(e)}")
            self.stats["failed_generations"] += 1
            return None

    def generate_oe_from_mcq(self, max_cases: Optional[int] = None):
        """Generate OE test sets from MCQ test sets"""
        
        # 1. Load MCQ test sets
        print("Loading MCQ test sets...")
        mcq_test_sets = self.load_mcq_test_sets()
        base_cases = mcq_test_sets["base"]
        golden_cases = mcq_test_sets["golden"]
        mix_cases = mcq_test_sets["mix"]

        # 2. 以golden为主，生成所有问题wording
        golden_question_map = {}
        
        def generate_question_task(golden_case):
            query = golden_case["original_query"]
            passage = golden_case["passage"]
            return query, self.generate_oe_question_only(query, passage)
        
        # 使用线程池并发生成问题
        with ThreadPoolExecutor(max_workers=2000) as executor:
            # 提交所有任务
            future_to_case = {
                executor.submit(generate_question_task, golden_case): golden_case 
                for i, golden_case in enumerate(golden_cases) 
                if not max_cases or i < max_cases
            }
            
            # 收集结果
            for future in tqdm(as_completed(future_to_case), 
                             total=len(future_to_case), 
                             desc="[问题生成]", 
                             unit="case"):
                try:
                    query, oe_question = future.result()
                    golden_question_map[query] = oe_question
                except Exception as e:
                    print(f"Error generating question: {str(e)}")
                    continue

        # 3. 用同一个问题，分别生成base/golden/mix的answer
        oe_test_sets = {"base": [], "golden": [], "mix": []}
        
        def generate_answers_task(item):
            query, oe_question = item
            
            # 查找对应的cases
            base_case = next((c for c in base_cases if c["original_query"] == query), None)
            golden_case = next((c for c in golden_cases if c["original_query"] == query), None)
            mix_case = next((c for c in mix_cases if c["original_query"] == query), None)
            
            results = []
            
            # 生成base答案
            if base_case:
                base_answer = self.generate_oe_answer_only(oe_question, passage=None)
                base_quality = self.assess_question_quality(
                    oe_question["question"],
                    base_answer["answer"],
                    "gpt-4o-mini"
                )
                results.append(("base", {
                    "original_query": base_case["original_query"],
                    "setting": "base",
                    "metadata": {
                        "source": "mcq_generated",
                        "setting": "base",
                        "has_passage": False,
                        "quality_score": base_quality["final_score"],
                        "generated_quality": base_quality,
                        "difficulty_level": "easy" if base_quality["difficulty"] <= 3 else "hard",
                        "is_valid": base_quality["is_valid"]
                    },
                    "open_ended": {
                        "gpt40": {
                            "content": {
                                "question": oe_question["question"],
                                "answer": base_answer["answer"],
                                "key_points": base_answer["key_points"],
                                "reasoning": base_answer["reasoning"]
                            }
                        }
                    }
                }))
            
            # 生成golden答案
            if golden_case:
                golden_answer = self.generate_oe_answer_only(oe_question, passage=golden_case["passage"])
                golden_quality = self.assess_question_quality(
                    oe_question["question"],
                    golden_answer["answer"],
                    "gpt-4o-mini"
                )
                results.append(("golden", {
                    "original_query": golden_case["original_query"],
                    "setting": "golden",
                    "metadata": {
                        "source": "mcq_generated",
                        "setting": "golden",
                        "has_passage": True,
                        "quality_score": golden_quality["final_score"],
                        "generated_quality": golden_quality,
                        "difficulty_level": "easy" if golden_quality["difficulty"] <= 3 else "hard",
                        "is_valid": golden_quality["is_valid"]
                    },
                    "open_ended": {
                        "gpt40": {
                            "content": {
                                "question": oe_question["question"],
                                "answer": golden_answer["answer"],
                                "key_points": golden_answer["key_points"],
                                "reasoning": golden_answer["reasoning"]
                            }
                        }
                    },
                    "passage": golden_case["passage"]
                }))
            
            # 生成mix答案
            if mix_case:
                # 检查mix是否与golden完全一致
                if golden_case and mix_case["passage"] == golden_case["passage"]:
                    # 如果passage完全一致，直接复制golden的答案和quality
                    print(f"Mix case identical to golden for query: {query[:50]}..., copying answer")
                    mix_answer = golden_answer
                    mix_quality = golden_quality
                else:
                    # 如果passage不同，重新生成
                    mix_answer = self.generate_oe_answer_only(oe_question, passage=mix_case["passage"])
                    mix_quality = self.assess_question_quality(
                        oe_question["question"],
                        mix_answer["answer"],
                        "gpt-4o-mini"
                    )
                
                results.append(("mix", {
                    "original_query": mix_case["original_query"],
                    "setting": "mix",
                    "metadata": {
                        "source": "mcq_generated",
                        "setting": "mix",
                        "has_passage": True,
                        "quality_score": mix_quality["final_score"],
                        "generated_quality": mix_quality,
                        "difficulty_level": "easy" if mix_quality["difficulty"] <= 3 else "hard",
                        "is_valid": mix_quality["is_valid"]
                    },
                    "open_ended": {
                        "gpt40": {
                            "content": {
                                "question": oe_question["question"],
                                "answer": mix_answer["answer"],
                                "key_points": mix_answer["key_points"],
                                "reasoning": mix_answer["reasoning"]
                            }
                        }
                    },
                    "passage": mix_case["passage"]
                }))
            
            return results
        
        # 使用线程池并发生成答案
        with ThreadPoolExecutor(max_workers=2000) as executor:
            # 提交所有任务
            future_to_item = {
                executor.submit(generate_answers_task, item): item 
                for item in golden_question_map.items()
            }
            
            # 收集结果
            for future in tqdm(as_completed(future_to_item), 
                             total=len(future_to_item), 
                             desc="[答案生成]", 
                             unit="case"):
                try:
                    results = future.result()
                    for setting, case in results:
                        oe_test_sets[setting].append(case)
                except Exception as e:
                    print(f"Error generating answers: {str(e)}")
                    continue

        # 4. 保存
        self.test_set = []
        for cases in oe_test_sets.values():
            self.test_set.extend(cases)
        self.save_test_set()
        self.print_statistics()

    def print_statistics(self):
        """Print generation statistics"""
        print(f"\n=== OE Generation Statistics ===")
        print(f"Total MCQ cases processed: {self.stats['total_mcq_cases']}")
        print(f"Successful OE generations: {self.stats['successful_oe_generations']}")
        print(f"Failed generations: {self.stats['failed_generations']}")
        print(f"Skipped cases: {self.stats['skipped_cases']}")
        
        if self.stats['total_mcq_cases'] > 0:
            success_rate = (self.stats['successful_oe_generations'] / self.stats['total_mcq_cases']) * 100
            print(f"Success rate: {success_rate:.2f}%")
        
        # Print quality statistics
        if self.stats["quality_scores"]["final_scores"]:
            print(f"\n=== Quality Statistics ===")
            final_scores = self.stats["quality_scores"]["final_scores"]
            print(f"Average final score: {sum(final_scores) / len(final_scores):.2f}")
            print(f"Min final score: {min(final_scores):.2f}")
            print(f"Max final score: {max(final_scores):.2f}")
            
            valid_count = sum(1 for score in final_scores if score >= 4.0)
            print(f"Valid questions (score >= 4.0): {valid_count}/{len(final_scores)} ({valid_count/len(final_scores)*100:.1f}%)")
            
            # Print average scores for each metric
            for metric in ["clarity", "difficulty", "relevance", "educational_value", "answer_quality", "cognitive_level", "answerability"]:
                scores = self.stats["quality_scores"][metric]
                if scores:
                    avg_score = sum(scores) / len(scores)
                    print(f"Average {metric}: {avg_score:.2f}")

    def save_test_set(self):
        """Save the generated test set to multiple JSON files (like MCQ)"""
        if not self.test_set:
            print("No test set to save")
            return
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 1. save simple and detailed versions by setting
        for setting in ["base", "golden", "mix"]:
            # get all cases for the setting
            cases = [case for case in self.test_set if case["setting"] == setting]
            
            # save simple version
            simple_cases = []
            for case in cases:
                simple_case = {
                    "original_query": case["original_query"],
                    "open_ended": {
                        "gpt40": {
                            "content": case["open_ended"]["gpt40"]["content"]
                        }
                    }
                }
                # only golden and mix cases have passage
                if setting in ["golden", "mix"]:
                    simple_case["passage"] = case["passage"]
                simple_cases.append(simple_case)
            
            simple_file = os.path.join(self.output_dir, f"test_set_{setting}_simple.json")
            with open(simple_file, 'w', encoding='utf-8') as f:
                json.dump(simple_cases, f, indent=2, ensure_ascii=False)
            
            # save detailed version
            detailed_file = os.path.join(self.output_dir, f"test_set_{setting}_detailed.json")
            with open(detailed_file, 'w', encoding='utf-8') as f:
                json.dump(cases, f, indent=2, ensure_ascii=False)
        
        # 2. save checkpoint (for resuming)
        checkpoint = {
            "total_cases": len(self.test_set),
            "test_set": self.test_set
        }
        checkpoint_file = os.path.join(self.output_dir, "test_set_checkpoint.json")
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False)
        
        # 3. save complete version (include all metadata)
        complete_test_set = {
            "metadata": {
                "source": "generated_from_mcq",
                "total_cases": len(self.test_set),
                "settings": ["base", "golden", "mix"],
                "statistics": self.stats
            },
            "test_cases": {
                "base": [case for case in self.test_set if case["setting"] == "base"],
                "golden": [case for case in self.test_set if case["setting"] == "golden"],
                "mix": [case for case in self.test_set if case["setting"] == "mix"]
            }
        }
        complete_file = os.path.join(self.output_dir, "test_set_complete.json")
        with open(complete_file, 'w', encoding='utf-8') as f:
            json.dump(complete_test_set, f, indent=2, ensure_ascii=False)
        
        # 4. save running record
        # get all complete queries (each query should have 3 cases)
        queries = {}
        for case in self.test_set:
            query = case["original_query"]
            if query not in queries:
                queries[query] = set()
            queries[query].add(case["setting"])
        
        # only queries with base, golden, mix versions are considered complete
        completed_queries = {query for query, settings in queries.items() if len(settings) == 3}
        
        # read existing record
        record_file = os.path.join(self.output_dir, "test_set_record.json")
        try:
            with open(record_file, 'r', encoding='utf-8') as f:
                record = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            record = {"completed_queries": []}
        
        # add new completed queries
        record["completed_queries"].extend(list(completed_queries))
        record["completed_queries"] = list(set(record["completed_queries"]))  # deduplicate
        
        # save record
        with open(record_file, 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        
        # calculate statistics
        base_cases = len([case for case in self.test_set if case["setting"] == "base"])
        golden_cases = len([case for case in self.test_set if case["setting"] == "golden"])
        mix_cases = len([case for case in self.test_set if case["setting"] == "mix"])
        
        # update self.stats
        self.stats["total_queries"] = len(completed_queries)
        self.stats["generated_questions"] = len(self.test_set)
        self.stats["skipped_queries"] = len(queries) - len(completed_queries)
        
        print(f"\nTest sets saved:")
        print("1. Simple versions:")
        for setting in ["base", "golden", "mix"]:
            print(f"   - {setting}: {os.path.join(self.output_dir, f'test_set_{setting}_simple.json')}")
        print("2. Detailed versions:")
        for setting in ["base", "golden", "mix"]:
            print(f"   - {setting}: {os.path.join(self.output_dir, f'test_set_{setting}_detailed.json')}")
        print(f"3. Checkpoint: {checkpoint_file}")
        print(f"4. Complete version: {complete_file}")
        print(f"5. Record file: {record_file}")
        print(f"\nTotal test cases: {len(self.test_set)}")
        print(f"Completed queries: {len(completed_queries)}")
        print(f"Base cases: {base_cases}")
        print(f"Golden cases: {golden_cases}")
        print(f"Mix cases: {mix_cases}")
        print(f"\nTest Set Statistics:")
        print(f"Total test cases: {len(self.test_set)}")
        print(f"Base cases: {base_cases}")
        print(f"Golden cases: {golden_cases}")
        print(f"Mix cases: {mix_cases}")
        print(f"\nTotal queries processed: {self.stats['total_queries']}")
        print(f"Queries skipped: {self.stats['skipped_queries']}")
        print(f"Questions generated: {self.stats['generated_questions']}")

    def assess_question_quality(self, question: str, answer: str, model: str) -> Dict:
        """Assess the quality of the open-ended question and answer (for RAG benchmark)"""
        prompt = f"""
        Evaluate the following open-ended question and answer pair:
        Q: {question}
        A: {answer}

        Please rate each aspect on a scale of 1–5 (where 1 is lowest and 5 is highest):

        1. Clarity: How clear and well-formulated is the question?
        2. Difficulty: How challenging is the question?
        3. Relevance: How relevant is the question to the reference document?
        4. Educational Value: How well does it test understanding of the reference document?
        5. Answer Quality: How comprehensive, accurate, and well-structured is the answer?
        6. Cognitive Level: Does it test higher-order thinking?
        7. Answerability: Can the question be answered using only the reference document?

        Return the response in JSON format:
        {{
            "clarity": 5,
            "difficulty": 3,
            "relevance": 4,
            "educational_value": 4,
            "answer_quality": 5,
            "cognitive_level": 4,
            "answerability": 5,
            "explanation": "..."
        }}
        """

        try:
            with self.api_semaphore:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that evaluates question quality. Always respond with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={ "type": "json_object" }
                )
            
            # Try to parse the response
            try:
                quality_scores = json.loads(response.choices[0].message.content)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON response: {str(e)}")
                print(f"Raw response: {response.choices[0].message.content}")
                return {
                    "clarity": 3,
                    "difficulty": 3,
                    "relevance": 3,
                    "educational_value": 3,
                    "answer_quality": 3,
                    "cognitive_level": 3,
                    "answerability": 3,
                    "explanation": f"Error parsing JSON response: {str(e)}",
                    "final_score": 3.0,
                    "is_valid": False
                }

            # evaluate the quality of the question  
            required_keys = ["clarity", "difficulty", "relevance", "educational_value", "answer_quality", "cognitive_level", "answerability", "explanation"]
            for key in required_keys:
                if key not in quality_scores:
                    raise ValueError(f"Missing key: {key}")
                if key != "explanation":
                    if not isinstance(quality_scores[key], (int, float)) or not 1 <= quality_scores[key] <= 5:
                        quality_scores[key] = 3  # fallback to 3
                        quality_scores["explanation"] += f"\n{key} score was invalid. Set to 3."

            # add final score - weighted average of all quality metrics
            weights = {
                "clarity": 0.20,
                "relevance": 0.20,
                "answer_quality": 0.20,
                "answerability": 0.20,
                "cognitive_level": 0.10,
                "educational_value": 0.10
            }
            
            quality_scores["final_score"] = round(
                sum(quality_scores[metric] * weight 
                    for metric, weight in weights.items()), 2)

            # add is_valid to check if the question is valid for benchmark
            # A question is valid if it meets minimum quality thresholds
            quality_scores["is_valid"] = (
                quality_scores["clarity"] >= 4 and
                quality_scores["relevance"] >= 4 and
                quality_scores["answer_quality"] >= 4 and
                quality_scores["answerability"] >= 4 and
                quality_scores["final_score"] >= 4.0
            )

            # Add detailed quality assessment explanation
            quality_scores["quality_assessment"] = {
                "meets_thresholds": quality_scores["is_valid"],
                "strengths": [
                    metric for metric, score in quality_scores.items()
                    if metric in weights and score >= 4
                ],
                "areas_for_improvement": [
                    metric for metric, score in quality_scores.items()
                    if metric in weights and score < 4
                ],
                "overall_quality": "High" if quality_scores["final_score"] >= 4.5 else
                                 "Medium" if quality_scores["final_score"] >= 4.0 else
                                 "Low"
            }

            return quality_scores

        except Exception as e:
            print(f"Error during quality assessment: {str(e)}")
            return {
                "clarity": 3,
                "difficulty": 3,
                "relevance": 3,
                "educational_value": 3,
                "answer_quality": 3,
                "cognitive_level": 3,
                "answerability": 3,
                "explanation": f"Error during assessment: {str(e)}",
                "final_score": 3.0,
                "is_valid": False
            }

    def generate_with_retry(self, func, *args, max_retries: int = 3) -> Dict:
        """Generate with retry mechanism"""
        for i in range(max_retries):
            try:
                return func(*args)  # synchronous call
            except Exception as e:
                print(f"Error during question generation: {str(e)}")
                if i < max_retries - 1:
                    print(f"Retry {i+1}/{max_retries} due to: {str(e)}")
                    time.sleep(1)  # use time.sleep
                else:
                    raise Exception("Failed to generate a valid question after all retries")

def main():
    """Main function"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate OE test sets from MCQ')
    parser.add_argument('--test', action='store_true', help='Run in test mode (process only 5 cases)')
    parser.add_argument('--num_cases', type=int, default=None, help='Number of cases to process (default: all)')
    parser.add_argument('--setting', choices=['base', 'golden', 'mix', 'all'], default='all', 
                       help='Which setting to process (default: all)')
    
    args = parser.parse_args()
    
    # Configuration
    mcq_dir = "/home/shared/RAG_DATA/benchmark/MCQ/generated_test_sets"
    output_dir = "/home/shared/RAG_DATA/benchmark/OE/generated_test_sets"
    
    # Create generator
    generator = OEFromMCQGenerator(mcq_dir, output_dir)
    
    # Determine number of cases to process
    if args.test:
        num_cases = 5
        print("🧪 Running in TEST MODE - processing only 5 cases")
    elif args.num_cases:
        num_cases = args.num_cases
        print(f"📊 Processing {num_cases} cases")
    else:
        num_cases = None
        print("🚀 Processing ALL cases")
    
    # Generate OE test sets from MCQ
    generator.generate_oe_from_mcq(max_cases=num_cases)
    
    print(f"\n✅ Generation completed!")
    if args.test:
        print("🧪 Test mode completed. Check the output files to verify everything works correctly.")
        print("💡 To process more cases, use: python generate_oe_from_mcq.py --num_cases 100")
        print("💡 To process all cases, use: python generate_oe_from_mcq.py")

if __name__ == "__main__":
    main() 