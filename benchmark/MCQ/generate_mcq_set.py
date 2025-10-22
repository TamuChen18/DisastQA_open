import json
import os
from typing import List, Dict, Set, Optional, Tuple
from openai import OpenAI
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk import word_tokenize, pos_tag
import time
import random
from dotenv import load_dotenv
import copy
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from nltk.tokenize import sent_tokenize

def load_json_file(file_path: str) -> Dict:
    """Load a JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON file {file_path}: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise

def save_json_file(file_path: str, data: Dict):
    """Save data to a JSON file"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving JSON file {file_path}: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise

class TestSetGenerator:
    def __init__(self, data_prepare_dir: str, output_dir: str):
        """Initialize test set generator"""
        self.data_prepare_dir = data_prepare_dir
        self.output_dir = output_dir
        self.processed_queries = set()
        self.test_cases_buffer = {
            'base': [],
            'golden': [],
            'mix': []
        }
        self.test_set = []
        self.question_types = ["multiple_choice", "open_ended"]
        self.task_types = ["qa", "qadoc", "factcheck", "twitter", "nli"]
        self.model_types = ["gpt40"]
        self.retrieval_types = ["base", "golden", "mix"]
        
        # load .env file from benchmark directory (parent directory)
        env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
        load_dotenv(dotenv_path=env_path)
        
        # use synchronous client
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Create semaphore for API rate limiting
        self.api_semaphore = threading.Semaphore(500)  # if the API allows, you can increase the number
        
        # Create locks for thread safety
        self.file_lock = threading.Lock()
        self.stats_lock = threading.Lock()
        
        # Statistics counters
        self.quality_passed = 0
        
        # Initialize filtered_queries as empty list
        self.filtered_queries = []
        
        # add more detailed statistics  
        self.stats = {
            "total_queries": 0,
            "skipped_queries": 0,
            "generated_questions": 0,
            "quality_scores": {
                "clarity": [],
                "difficulty": [],
                "relevance": [],
                "educational_value": [],
                "option_quality": [],
                "cognitive_level": [],
                "answerability": []
            }
        }

        # add batch saving related variables
        self.batch_size = 50  # save 50 test cases at a time
        self.last_checkpoint_update = 0

        # add batch processing to optimize memory usage
        self.BATCH_SIZE = 1000  # process 1000 queries at a time

        # add retry mechanism to handle failed cases
        self.MAX_RETRIES = 3

    def initialize(self):
        """Initialize the generator by loading all queries"""
        try:
            # 1. load all data files
            data_files = [f for f in os.listdir(self.data_prepare_dir) if f.endswith('_by_score.json')]
            print(f"Found {len(data_files)} data files")
            
            # 2. merge all queries, ensure uniqueness
            all_queries = []
            unique_queries = set()  # for deduplication
            for data_file in data_files:
                data_path = os.path.join(self.data_prepare_dir, data_file)
                file_queries = load_json_file(data_path)
                if isinstance(file_queries, list):
                    for query in file_queries:
                        if query["user_query"] not in unique_queries:
                            unique_queries.add(query["user_query"])
                            all_queries.append(query)
                else:
                    print(f"Warning: {data_file} is not a list of queries")
            
            # 3. convert to dictionary format, key is query_id
            self.queries = {}
            for i, query in enumerate(all_queries):
                query_id = f"query_{i}"
                self.queries[query_id] = query
            
            print(f"Total unique queries loaded: {len(self.queries)}")
            
            # 4. create output directory
            os.makedirs(self.output_dir, exist_ok=True)
            
        except Exception as e:
            print(f"Error initializing generator: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise

    def load_test_queries(self, num_cases: int = 10) -> List[Dict]:
        all_queries = []
        unique_queries = set()  # for deduplication
        print("\nLoading test queries...")
        
        # Get all files from data_prepare directory
        files = [f for f in os.listdir(self.data_prepare_dir)
                if f.endswith('_by_score.json')]
        
        # Process one file at a time
        for filename in files:
            file_path = os.path.join(self.data_prepare_dir, filename)
            print(f"\nProcessing file: {filename}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"Total queries in file: {len(data)}")
                
                # Process each query
                for query in data:
                    user_query = query.get("user_query", "")
                    # only add unique user_query
                    if user_query and user_query not in unique_queries:
                        unique_queries.add(user_query)
                        # Keep essential fields
                        simplified_query = {
                            "user_query": user_query,
                            "filename": filename,
                            "general_type": query["general_type"],
                            "passages_by_score": query["passages_by_score"]
                        }
                        all_queries.append(simplified_query)
        
        print(f"\nTotal unique queries loaded: {len(all_queries)}")
        
        # randomly select num_cases queries
        if len(all_queries) > num_cases:
            selected_queries = random.sample(all_queries, num_cases)
            print(f"\nRandomly selected {num_cases} queries")
            return selected_queries
        else:
            print(f"\nNot enough queries, using all {len(all_queries)} queries")
            return all_queries

    def _process_single_query(self, query: Dict):
        """Process a single query and generate QA"""
        print(f"\nProcessing query: {query.get('user_query', '')[:100]}...")
        
        # Try to generate a high quality question with retries
        max_retries = 3
        for retry in range(max_retries):
            print(f"\nAttempt {retry + 1}/{max_retries}")
            
            # Randomly select a score=3 passage
            score_3_passages = query["passages_by_score"].get("3", [])
            if not score_3_passages:
                print("No score=3 passages found")
                return
            passage = random.choice(score_3_passages)
            print(f"Selected passage from {len(score_3_passages)} score=3 passages")
            
            # First assess the quality of the original query
            original_quality = self.assess_question_quality(
                query["user_query"],
                passage,
                "gpt-4o-mini"
            )
            print(f"Original query quality: {original_quality}")
            
            # Generate questions using GPT-4o-mini
            print("Generating multiple choice question...")
            gpt40_mc = self.generate_with_retry(self.generate_multiple_choice,
                query["user_query"], 
                passage,  # use the selected score=3 passage
                "gpt-4o-mini",
                query["general_type"]
            )
            
            if gpt40_mc is None or "content" not in gpt40_mc:
                print("Failed to generate multiple choice question")
                if retry < max_retries - 1:
                    print("Retrying question generation...")
                    continue
                else:
                    print("Max retries reached, skipping this query")
                    return
            
            # Assess GPT-4 generated questions
            correct_idx = ord(gpt40_mc["content"]["correct_option"].upper()) - ord("A")
            correct_answer = gpt40_mc["content"]["options"][correct_idx]
            gpt40_mc_quality = self.assess_question_quality(
                gpt40_mc["content"]["question"],
                correct_answer,
                "gpt-4o-mini"
            )
            
            # Create base test case
            base_case = {
                "original_query": query["user_query"],
                "setting": "base",
                "metadata": {
                    "query_id": query["filename"],
                    "passage_id": "",
                    "quality_score": 0,
                    "original_quality": original_quality,
                    "generated_quality": gpt40_mc_quality,
                    "difficulty_level": "easy" if original_quality["difficulty"] <= 3 else "hard"
                },
                "multiple_choice": {
                    "gpt40": {
                        "content": gpt40_mc["content"]
                    }
                }
            }
            self.test_set.append(base_case)
            print("Added base test case")

            # Generate test cases for each setting
            for setting in ["golden", "mix"]:
                print(f"\nGenerating test case for setting: {setting}")
                
                # Get passage based on setting
                if setting == "golden":
                    setting_passage = passage  # use the score=3 passage
                else:  # mix setting
                    # get low-score passages
                    low_score_passages = []
                    for score in range(3):  # 0,1,2
                        low_score_passages.extend(query["passages_by_score"].get(str(score), []))
                    
                    # generate
                    setting_passage = self._mix_passages(passage, query["passages_by_score"])
                
                # Create test case for this setting
                setting_test_case = {
                    "original_query": query["user_query"],
                    "passage": setting_passage,
                    "setting": setting,
                    "metadata": {
                        "query_id": query["filename"],
                        "passage_id": "",
                        "quality_score": 3 if setting == "golden" else 1.5,
                        "original_quality": original_quality,
                        "generated_quality": gpt40_mc_quality,
                        "difficulty_level": "easy" if original_quality["difficulty"] <= 3 else "hard"
                    },
                    "multiple_choice": {
                        "gpt40": {
                            "content": gpt40_mc["content"]
                        }
                    }
                }   
                self.test_set.append(setting_test_case)
            
            # Save after each successful query
            self.save_test_set()
            print(f"Saved test set with {len(self.test_set)} cases")
            
            # If we get here, we've successfully generated a test case
            break

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

    def generate_multiple_choice(self, user_query: str, positive_doc: str, model: str, general_type: str) -> Dict:
        """generate multiple choice questions"""
        prompt = f"""
        You are a helpful assistant that creates high-quality multiple-choice questions (MCQs) for evaluation and educational benchmarking.

        Given the following **user query** and **positive document** (the gold evidence), construct **one MCQ** with exactly 4 options:

        Task Type: {general_type}

        ---
        User Query:
        \"\"\"{user_query}\"\"\"

        Reference Document:
        \"\"\"{positive_doc}\"\"\"
        ---

        Instructions:

        1. Create a clear, concise multiple-choice **question** that can only be correctly answered using the reference document.
        2. Generate **one correct option** based on the document.
        3. Generate **three plausible but factually incorrect distractors**:
           - These distractors must be *semantically related* to the topic
           - They should be *plausible* but clearly incorrect
           - They should be at the *same level of specificity* as the correct answer
           - They should *not* be obviously wrong or unrelated
           - They should *not* be too similar to each other
        4. Do **not repeat the reference document verbatim**.
        5. Ensure all four options are grammatically and stylistically consistent.
        6. Randomize the position of the correct answer (A/B/C/D).
        7. For `reason`, briefly explain **why the correct option is correct** based on the document (in 1-2 sentences).
        8. Each option MUST start with its letter (A, B, C, or D) followed by a period and a space.

        Return only the following JSON format:

        {{
        "question": "...",
        "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
        "correct_option": "C",
        "reason": "..."
        }}
        """
        
        try:
            with self.api_semaphore:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that generates high-quality multiple choice questions. Always respond with valid JSON."},
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
                raise Exception("Failed to generate a valid question")
            
            # Validate the question data structure
            required_keys = ["question", "options", "correct_option", "reason"]
            if not all(key in question_data for key in required_keys):
                print(f"Missing required keys in question data: {question_data}")
                raise Exception("Failed to generate a valid question")
            
            if len(question_data["options"]) != 4:
                print(f"Invalid number of options: {len(question_data['options'])}")
                raise Exception("Failed to generate a valid question")
            
            if question_data["correct_option"] not in ["A", "B", "C", "D"]:
                print(f"Invalid correct option: {question_data['correct_option']}")
                raise Exception("Failed to generate a valid question")
            
            return {
                "content": question_data
            }
                
        except Exception as e:
            print(f"Error during question generation: {str(e)}")
            raise Exception("Failed to generate a valid question")

    # def generate_open_ended(self, user_query: str, positive_doc: str, model: str) -> Dict:
    #     """generate open ended questions"""
    #     prompt = f"""
    #     Based on the following question and reference answer, generate a detailed answer.
        
    #     Question: {user_query}
    #     Reference Answer: {positive_doc}
        
    #     Generate a comprehensive answer that addresses the question.
    #     """
        
    #     response = self.client.chat.completions.create(
    #         model=model,
    #         messages=[
    #             {"role": "system", "content": "You are a helpful assistant that generates detailed answers."},
    #             {"role": "user", "content": prompt}
    #         ]
    #     )
        
    #     return {
    #         "question": user_query,
    #         "reference_answer": positive_doc,
    #         "generated_answer": response.choices[0].message.content
    #     }

    def get_documents_for_query(self, query: Dict, retrieval_type: str, top_k: int = 5) -> List[Dict]:
        """Get documents for query based on retrieval type"""
        passages_by_score = query.get('passages_by_score', {})
        
        if retrieval_type == 'base':
            return []  # base case does not need documents
            
        elif retrieval_type == 'golden':
            # only use score=3 documents
            score_3_passages = passages_by_score.get('3', [])
            return score_3_passages[:top_k]
            
        elif retrieval_type == 'mix':
            # use score=3 documents, but replace 30% with low-score documents
            score_3_passages = passages_by_score.get('3', [])
            if not score_3_passages:
                return []
                
            # get all passages
            all_passages = []
            for score in range(4):  # 0, 1, 2, 3
                all_passages.extend(passages_by_score.get(str(score), []))
            
            # randomly select 30% of passages as mix version
            num_passages = len(all_passages)
            num_to_select = int(num_passages * 0.3)  # 30% of passages
            if num_to_select > 0:
                mix_passage = random.sample(all_passages, num_to_select)
            else:
                mix_passage = score_3_passages[0]
            
            return [{"passage": mix_passage}]
        
        return []


    
    def generate_test_set(self, num_cases: int = 10) -> List[Dict]:
        """Generate a test set with specified number of cases"""
        # 1. get all queries
        all_queries = list(self.queries.values())
        if not all_queries:
            print("Warning: No queries found")
            return []
        
        # 2. randomly select specified number of queries
        selected_queries = random.sample(all_queries, min(num_cases, len(all_queries)))
        print(f"Randomly selected {len(selected_queries)} queries")
        
        # 3. generate test case for each query
        test_set = []  # store all test cases
        processed_queries = set()  # store successfully processed queries
        skipped_queries = []  # store skipped queries
        
        # use tqdm to show progress bar
        with tqdm(total=num_cases, desc="Generating test cases") as pbar:
            for query in selected_queries:
                # 4. generate test case
                test_case = self._generate_test_case(query, query["passages_by_score"])
                if test_case:
                    # add to test set
                    test_set.extend([
                        test_case["golden"],
                        test_case["base"],
                        test_case["mix"]
                    ])
                    processed_queries.add(query["user_query"])
                    print(f"Successfully generated test case for query: {query['user_query']}")
                    pbar.update(1)  # only update progress bar when successfully processed query
                else:
                    skipped_queries.append(query["user_query"])
                    print(f"Skipped query: {query['user_query']}")
                
                # if reach the target number, stop
                if len(processed_queries) >= num_cases:
                    break
        
        # 5. randomly select 30% of mix cases for passage replacement
        mix_cases = [case for case in test_set if case["setting"] == "mix"]
        num_to_replace = int(len(mix_cases) * 0.3)
        replace_indices = random.sample(range(len(mix_cases)), num_to_replace)
        
        # 6. replace passage for selected cases
        for i in replace_indices:
            case = mix_cases[i]
            query = case["original_query"]
            # find the corresponding query
            query_data = next(q for q in selected_queries if q["user_query"] == query)
            
            # replace passage
            mix_passage, mix_quality_score, replacement_info = self._mix_passages(
                case["passage"],
                query_data["passages_by_score"],
                should_replace=True
            )
            
            # update mix case
            case["passage"] = mix_passage
            case["metadata"]["quality_score"] = mix_quality_score
            case["metadata"]["replacement_info"] = replacement_info
        
        # 7. print statistics
        print(f"\nProcessed {len(processed_queries)} queries successfully")
        print(f"Skipped {len(skipped_queries)} queries")
        if skipped_queries:
            print("\nSkipped queries:")
            for query in skipped_queries:
                print(f"- {query}")
        
        return test_set

    def assess_question_quality(self, question: str, answer: str, model: str) -> Dict:
        """Assess the quality of the MCQ (for RAG benchmark)"""
        prompt = f"""
        Evaluate the following question and answer pair:
        Q: {question}
        A: {answer}

        Please rate each aspect on a scale of 1–5 (where 1 is lowest and 5 is highest):

        1. Clarity: How clear and well-formulated is the question?
        2. Difficulty: How challenging is the question?
        3. Relevance: How relevant is the question to the reference document?
        4. Educational Value: How well does it test understanding of the reference document?
        5. Option Quality: How plausible and well-distinguished are the distractors?
        6. Cognitive Level: Does it test higher-order thinking?
        7. Answerability: Can the question be answered using only the reference document?

        Return the response in JSON format:
        {{
            "clarity": 5,
            "difficulty": 3,
            "relevance": 4,
            "educational_value": 4,
            "option_quality": 5,
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
                    "option_quality": 3,
                    "cognitive_level": 3,
                    "answerability": 3,
                    "explanation": f"Error parsing JSON response: {str(e)}",
                    "final_score": 3.0,
                    "is_valid": False
                }

            # evaluate the quality of the question  
            required_keys = ["clarity", "difficulty", "relevance", "educational_value", "option_quality", "cognitive_level", "answerability", "explanation"]
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
                "option_quality": 0.20,
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
                quality_scores["option_quality"] >= 4 and
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
                "option_quality": 3,
                "cognitive_level": 3,
                "answerability": 3,
                "explanation": f"Error during assessment: {str(e)}",
                "final_score": 3.0,
                "is_valid": False
            }

    def save_test_case_sync(self, test_case: Dict, query_id: str, processed_queries: Set):
        try:
            # ensure output directory exists
            os.makedirs(self.output_dir, exist_ok=True)
            
            # add test case to buffer
            setting = test_case["setting"]
            self.test_cases_buffer[setting].append(test_case)
            
            # update processed queries
            processed_queries.add(query_id)
            
            # if buffer reaches batch size, save to file
            if len(self.test_cases_buffer[setting]) >= self.batch_size:
                self._save_batch(setting)
            
            # update checkpoint periodically
            current_count = len(processed_queries)
            if current_count - self.last_checkpoint_update >= self.batch_size:
                self._update_checkpoint(processed_queries)
                self.last_checkpoint_update = current_count
                
            # save after processing 100 queries
            if len(processed_queries) % 100 == 0:
                self.save_checkpoint()
                
        except Exception as e:
            print(f"Error saving test case: {str(e)}")
            print(f"Error details: {type(e).__name__}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            raise

    def _save_batch(self, setting: str):
        """Save buffered test cases to file"""
        if not self.test_cases_buffer[setting]:
            return
        
        try:
            # read existing data
            file_path = os.path.join(self.output_dir, f'{setting}_test_set.json')
            existing_cases = []
            
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if content.strip():  # ensure file is not empty
                            existing_cases = json.loads(content)
                except json.JSONDecodeError as e:
                    print(f"Warning: Error reading existing file for {setting}, starting fresh: {str(e)}")
                    existing_cases = []
            
            # add new data
            all_cases = existing_cases + self.test_cases_buffer[setting]
            
            # save full test set
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(json.dumps(all_cases, ensure_ascii=False, indent=2))
            
            # save lightweight version
            lite_path = os.path.join(self.output_dir, f'{setting}_test_set_lite.json')
            lite_cases = [{
                'question': case['original_query'],
                'passages': case['retrieved_documents']
            } for case in all_cases]
            
            with open(lite_path, 'w', encoding='utf-8') as f:
                f.write(json.dumps(lite_cases, ensure_ascii=False, indent=2))
            
            # clear buffer
            self.test_cases_buffer[setting] = []
            
        except Exception as e:
            print(f"Error saving batch for {setting}: {str(e)}")
            print(f"Error details: {type(e).__name__}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            raise

    def _update_checkpoint(self, processed_queries: Set):
        """update checkpoint file"""
        try:
            checkpoint_file = os.path.join(self.output_dir, 'checkpoint.json')
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "processed_queries": list(processed_queries),
                    "stats": self.stats
                }, f, indent=2, ensure_ascii=False)
            print(f"Checkpoint saved to {checkpoint_file}")
        except Exception as e:
            print(f"Error saving checkpoint: {str(e)}")
            raise

    def _mix_passages(self, best_passage: str, passages_by_score: Dict, should_replace: bool = False) -> Tuple[str, float, Dict]:
        """Mix passages for a query"""
        if not should_replace:
            return best_passage, 3.0, {
                "is_replaced": False,
                "reason": "not_selected_for_replacement"
            }
        
        # 1. find all low-score passages (score 0, 1, 2)
        low_score_passages = {}
        for score in ["0", "1", "2"]:
            if score in passages_by_score and passages_by_score[score]:
                low_score_passages[score] = passages_by_score[score]
        
        if not low_score_passages:
            print(f"Warning: No low score passages found for replacement")
            return best_passage, 3.0, {
                "is_replaced": False,
                "reason": "no_low_score_passages"
            }
        
        # 2. count the number of passages for each score
        score_counts = {score: len(passages) for score, passages in low_score_passages.items()}
        available_scores = list(score_counts.keys())
        
        # 3. calculate weights, make the probability of selecting each score equal
        inverse_weights = [1.0 / score_counts[score] for score in available_scores]
        total_inverse_weight = sum(inverse_weights)
        weights = [w / total_inverse_weight for w in inverse_weights]
        
        # 4. randomly select a score based on weights
        selected_score = random.choices(available_scores, weights=weights, k=1)[0]
        
        # 5. randomly select a passage from the selected score
        selected_passage = random.choice(low_score_passages[selected_score])
        
        print(f"Replacing score=3 passage with score={selected_score} passage")
        
        return selected_passage, float(selected_score), {
            "is_replaced": True,
            "reason": "randomly_selected_low_score_passage",
            "original_score": 3.0,
            "new_score": float(selected_score)
        }

    def _load_passages(self, query_id: str, score: str) -> List[str]:
        """Load passages for a query and score"""
        try:
            # get passages by score directly from passages_by_score
            return self.queries[query_id]["passages_by_score"].get(score, [])
        except Exception as e:
            print(f"Error loading passages for query {query_id} and score {score}: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return []

    def _load_passages_by_score(self, query_id: str) -> Dict:
        """Load passages by score for a query"""
        try:
            return self.queries[query_id]["passages_by_score"]
        except Exception as e:
            print(f"Error loading passages by score for query_id {query_id}: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return {}

    def _generate_test_case(self, query: Dict, passages_by_score: Dict) -> Optional[Dict]:
        """Generate a test case for a query"""
        try:
            # 1. get query information
            user_query = query["user_query"]
            general_type = query["general_type"]
            
            # 2. check if there are passages
            if not passages_by_score:
                print(f"Warning: No passages found for query")
                return None
            
            # 3. find all score=3 passages
            score_3_passages = passages_by_score.get("3", [])
            if not score_3_passages:
                print(f"Warning: No score=3 passages found for query")
                return None
            
            # 4. assess the quality of score=3 passages, select the best one
            best_passage = None
            best_quality = 0
            best_quality_scores = None
            for passage in score_3_passages:
                with self.api_semaphore:
                    quality = self.assess_question_quality(
                        user_query,
                        passage,
                        "gpt-4o-mini"
                    )
                if quality["final_score"] > best_quality:
                    best_quality = quality["final_score"]
                    best_passage = passage
                    best_quality_scores = quality
            
            # if no best passage is found, use the first score=3 passage
            if not best_passage:
                print(f"Warning: No best passage found, using first score=3 passage")
                best_passage = score_3_passages[0]
                with self.api_semaphore:
                    best_quality_scores = self.assess_question_quality(
                        user_query,
                        best_passage,
                        "gpt-4o-mini"
                    )
            
            # 5. generate
            with self.api_semaphore:
                mc_question = self.generate_multiple_choice(
                    user_query,
                    best_passage,
                    "gpt-4o-mini",
                    general_type
                )
            
            if not mc_question:
                print(f"Warning: Failed to generate multiple choice question")
                return None
            
            # 6. generate golden case
            golden_case = {
                "original_query": user_query,
                "passage": best_passage,
                "setting": "golden",
                "metadata": {
                    "general_type": general_type,
                    "quality_score": 3.0,
                    "passage_quality": best_quality_scores,
                    "difficulty_level": "easy" if best_quality_scores["difficulty"] <= 3 else "hard"
                },
                "multiple_choice": {
                    "gpt40": {
                        "content": mc_question["content"]
                    }
                }
            }
            
            # 7. generate base case (without passage)
            base_case = {
                "original_query": user_query,
                "setting": "base",
                "metadata": {
                    "general_type": general_type,
                    "quality_score": 0.0,
                    "passage_quality": best_quality_scores,
                    "difficulty_level": "easy" if best_quality_scores["difficulty"] <= 3 else "hard"
                },
                "multiple_choice": {
                    "gpt40": {
                        "content": mc_question["content"]
                    }
                }
            }
            
            # 8. generate mix case (use golden passage first)
            mix_case = {
                "original_query": user_query,
                "passage": best_passage,
                "setting": "mix",
                "metadata": {
                    "general_type": general_type,
                    "quality_score": 3.0,
                    "passage_quality": best_quality_scores,
                    "difficulty_level": "easy" if best_quality_scores["difficulty"] <= 3 else "hard",
                    "replacement_info": {
                        "is_replaced": False,
                        "reason": "not_selected_for_replacement"
                    }
                },
                "multiple_choice": {
                    "gpt40": {
                        "content": mc_question["content"]
                    }
                }
            }
            
            # 9. ensure all cases are generated successfully
            if not all([golden_case, base_case, mix_case]):
                print(f"Warning: Failed to generate all cases")
                return None
                
            return {
                "golden": golden_case,
                "base": base_case,
                "mix": mix_case
            }
            
        except Exception as e:
            print(f"Error generating test case: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return None

    def generate_test_set_for_file(self, data_file: str, num_cases: Optional[int] = None) -> List[Dict]:
        """Generate a test set for a specific file"""
        try:
            # 1. load all queries from the specified file
            data_path = os.path.join(self.data_prepare_dir, data_file)
            file_queries = load_json_file(data_path)
            
            if not file_queries:
                print(f"Warning: No queries found in {data_file}")
                return []
            
            # 2. randomly select specified number of queries
            if num_cases is None:
                selected_queries = file_queries  # if num_cases is None, use all queries
            else:
                selected_queries = random.sample(file_queries, min(num_cases, len(file_queries)))
            print(f"Selected {len(selected_queries)} queries from {data_file}")
            
            # 3. use thread pool to process queries
            test_set = []  # store all test cases
            processed_queries = set()  # store successfully processed queries
            skipped_queries = []  # store skipped queries
            
            # create thread lock to protect shared resources
            test_set_lock = threading.Lock()
            processed_lock = threading.Lock()
            skipped_lock = threading.Lock()
            
            def process_query(query):
                try:
                    test_case = self._generate_test_case(query, query["passages_by_score"])
                    if test_case:
                        with test_set_lock:
                            test_set.extend([
                                test_case["golden"],
                                test_case["base"],
                                test_case["mix"]
                            ])
                        with processed_lock:
                            processed_queries.add(query["user_query"])
                        print(f"Successfully generated test case for query: {query['user_query']}")
                        return True
                    else:
                        with skipped_lock:
                            skipped_queries.append(query["user_query"])
                        print(f"Skipped query: {query['user_query']}")
                        return False
                except Exception as e:
                    print(f"Error processing query {query['user_query']}: {str(e)}")
                    with skipped_lock:
                        skipped_queries.append(query["user_query"])
                    return False
            
            # use thread pool to process queries
            with ThreadPoolExecutor(max_workers=300) as executor:
                # submit all tasks
                futures = [executor.submit(process_query, query) for query in selected_queries]
                
                # use tqdm to show progress
                with tqdm(total=len(selected_queries), desc=f"Processing queries from {data_file}") as pbar:
                    for future in as_completed(futures):
                        pbar.update(1)
            
            # 4. randomly select 30% of mix cases for passage replacement
            mix_cases = [case for case in test_set if case["setting"] == "mix"]
            num_to_replace = int(len(mix_cases) * 0.3)
            replace_indices = random.sample(range(len(mix_cases)), num_to_replace)
            
            # 5. replace passage for selected cases
            for i in replace_indices:
                case = mix_cases[i]
                query = case["original_query"]
                # find the corresponding query
                query_data = next(q for q in selected_queries if q["user_query"] == query)
                
                # replace passage
                mix_passage, mix_quality_score, replacement_info = self._mix_passages(
                    case["passage"],
                    query_data["passages_by_score"],
                    should_replace=True
                )
                
                # update mix case
                case["passage"] = mix_passage
                case["metadata"]["quality_score"] = mix_quality_score
                case["metadata"]["replacement_info"] = replacement_info
            
            # 6. print statistics
            print(f"\nProcessed {len(processed_queries)} queries successfully")
            print(f"Skipped {len(skipped_queries)} queries")
            if skipped_queries:
                print("\nSkipped queries:")
                for query in skipped_queries:
                    print(f"- {query}")
            
            return test_set
            
        except Exception as e:
            print(f"Error generating test set for {data_file}: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return []

    def save_test_set(self):
        """Save the generated test set to multiple JSON files"""
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
                    "multiple_choice": {
                        "gpt40": {
                            "content": case["multiple_choice"]["gpt40"]["content"]
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

    def save_checkpoint(self):
        """Save checkpoint to file"""
        try:
            checkpoint_file = os.path.join(self.output_dir, 'checkpoint.json')
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "processed_queries": list(self.processed_queries),
                    "stats": self.stats
                }, f, indent=2, ensure_ascii=False)
            print(f"Checkpoint saved to {checkpoint_file}")
        except Exception as e:
            print(f"Error saving checkpoint: {str(e)}")
            raise

    def print_statistics(self):
        """Print statistics about the generated test set"""
        if not self.test_set:
            print("No test set to analyze")
            return
        
        print("\nTest Set Statistics:")
        print(f"Total test cases: {len(self.test_set)}")
        
        # Count cases by setting
        settings = ["base", "golden", "mix"]
        for setting in settings:
            count = len([case for case in self.test_set if case["setting"] == setting])
            print(f"{setting.capitalize()} cases: {count}")
        
        # Calculate average quality scores
        if self.stats["quality_scores"]["clarity"]:
            print("\nAverage Quality Scores:")
            for metric in self.stats["quality_scores"]:
                scores = self.stats["quality_scores"][metric]
                if scores:
                    avg_score = sum(scores) / len(scores)
                    print(f"{metric.capitalize()}: {avg_score:.2f}")
        
        # Print other statistics
        print(f"\nTotal queries processed: {self.stats['total_queries']}")
        print(f"Queries skipped: {self.stats['skipped_queries']}")
        print(f"Questions generated: {self.stats['generated_questions']}")

def main(num_cases: Optional[int] = None):
    """Main function to generate test set"""
    try:
        # 1. set directory
        data_prepare_dir = '/home/shared/RAG_DATA/benchmark/data_prepare'
        output_dir = '/home/shared/RAG_DATA/benchmark/generated_test_sets'
        
        # 2. get all data files
        data_files = [f for f in os.listdir(data_prepare_dir) if f.endswith('_by_score.json')]
        print(f"Found {len(data_files)} data files")
        
        # 3. create generator and load data
        generator = TestSetGenerator(data_prepare_dir, output_dir)
        generator.initialize()
        
        # 4. if num_cases is specified, only process the first file
        if num_cases is not None:
            data_files = [data_files[0]]
            print(f"\nProcessing only first file with {num_cases} cases...")
        
        # 5. use thread pool to process files
        with ThreadPoolExecutor(max_workers=8) as executor:
            # submit all tasks
            future_to_file = {
                executor.submit(generator.generate_test_set_for_file, data_file, num_cases): data_file 
                for data_file in data_files
            }
            
            # process results
            all_test_set = []
            for future in as_completed(future_to_file):
                data_file = future_to_file[future]
                try:
                    test_set = future.result()
                    all_test_set.extend(test_set)
                    print(f"Test set generated for {data_file}")
                except Exception as e:
                    print(f"Error processing {data_file}: {str(e)}")
                    import traceback
                    print(traceback.format_exc())
            
            # save all test set
            generator.test_set = all_test_set
            generator.save_test_set()
            
    except Exception as e:
        print(f"Error in main: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    # process all queries
    main(num_cases=None)