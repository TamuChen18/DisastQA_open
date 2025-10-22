#!/usr/bin/env python3
"""
Evaluate GPT4o Answers vs Human Answers

This script compares GPT4o-generated answers with human answers
to assess if GPT4o can replace human answers for OE evaluation.
"""

import json
import os
from typing import List, Dict
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import torch
from openai import OpenAI
from dotenv import load_dotenv
import time
from collections import Counter
import argparse

class GPT4oEvaluator:
    def __init__(self, gpt4o_answers_file: str, bleu_ngrams: List[int] = None):
        """
        Initialize the GPT4o evaluator
        
        Args:
            gpt4o_answers_file: Path to GPT4o answers dataset
            bleu_ngrams: List of n-gram orders to calculate BLEU for (default: [1, 2, 3, 4])
        """
        self.gpt4o_answers_file = gpt4o_answers_file
        self.bleu_ngrams = bleu_ngrams or [1, 2, 3, 4]  # Default to BLEU-1,2,3,4
        self.data = self.load_data()
        
        # Load environment variables
        env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
        load_dotenv(dotenv_path=env_path)
        
        # Initialize OpenAI client for quality assessment
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Initialize evaluation tools
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smooth = SmoothingFunction().method1
        
        # Check if CUDA is available for BERTScore
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        print(f"BLEU n-grams: {self.bleu_ngrams}")
        
        # Initialize SentenceTransformer globally
        self._init_sentence_transformer()
        
        # Initialize caches for efficiency
        self.rephrase_cache = {}
        self.semantic_cache = {}
        self.keypoint_cache = {}  # Cache for extracted keypoints

    def _init_sentence_transformer(self):
        """Initialize SentenceTransformer model globally"""
        try:
            from sentence_transformers import SentenceTransformer
            # Use all-mpnet-base-v2 for better quality (larger but more accurate)
            # Fallback to e5-small-v2 if memory is limited
            try:
                self.sentence_model = SentenceTransformer('all-mpnet-base-v2')
                print("Using all-mpnet-base-v2 for semantic similarity (higher quality)")
            except Exception as e:
                print(f"Failed to load all-mpnet-base-v2, falling back to e5-small-v2: {e}")
                self.sentence_model = SentenceTransformer('intfloat/e5-small-v2')
                print("Using e5-small-v2 for semantic similarity (faster)")
        except ImportError:
            print("Warning: sentence-transformers not available. Semantic similarity will be disabled.")
            self.sentence_model = None

    def load_data(self) -> List[Dict]:
        """Load GPT4o answers dataset"""
        with open(self.gpt4o_answers_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def extract_key_points(self, question: str, answer: str) -> List[str]:
        """Extract key points from answer using GPT4o with caching"""
        # Create cache key
        cache_key = f"{question[:100]}_{answer[:100]}"  # Use first 100 chars for key
        
        # Check cache first
        if cache_key in self.keypoint_cache:
            return self.keypoint_cache[cache_key]
        
        try:
            prompt = f"""You are a helpful assistant. Given an open-ended question and a reference answer, extract the most important key points (facts, definitions, conclusions) that any good answer should contain.

Your output should be a numbered list of 3 to 5 concise key points.

Question: {question}

Reference Answer: {answer}

Return format (JSON list):
[
  "key point 1",
  "key point 2", 
  "key point 3"
]

- Do not copy long paragraphs.
- Each key point must represent a distinct concept.
- Keep each point within 30 words."""

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at extracting key information from text."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.1
            )
            
            content = response.choices[0].message.content.strip()
            
            # Try to parse JSON
            try:
                # Remove markdown code blocks if present
                if content.startswith('```json'):
                    content = content[7:]
                if content.endswith('```'):
                    content = content[:-3]
                
                key_points = json.loads(content)
                # Cache the result
                self.keypoint_cache[cache_key] = key_points
                return key_points
            except json.JSONDecodeError:
                # Fallback: extract numbered points
                lines = content.split('\n')
                key_points = []
                for line in lines:
                    line = line.strip()
                    if line and (line[0].isdigit() or line.startswith('-')):
                        # Remove numbering and clean up
                        point = line.split('.', 1)[-1].strip()
                        if point.startswith('- '):
                            point = point[2:]
                        if point:
                            key_points.append(point)
                key_points = key_points[:5]  # Limit to 5 points
                # Cache the result
                self.keypoint_cache[cache_key] = key_points
                return key_points
        
        except Exception as e:
            print(f"Error extracting key points: {e}")
            return []

    def pre_extract_all_keypoints(self):
        """Pre-extract all keypoints to avoid API calls during evaluation"""
        print("Pre-extracting keypoints for all questions...")
        
        for i, item in enumerate(self.data):
            print(f"Extracting keypoints for item {i+1}/{len(self.data)}...")
            
            question = item.get('question', '')
            human_answer = item.get('human_answer', '')
            gpt4o_answer = item.get('gpt4o_answer', '')
            
            if question and human_answer:
                # Extract human keypoints
                human_keypoints = self.extract_key_points(question, human_answer)
            
            if question and gpt4o_answer:
                # Extract GPT4o keypoints
                gpt4o_keypoints = self.extract_key_points(question, gpt4o_answer)
            
            # Add delay to avoid rate limiting
            time.sleep(0.5)
        
        print(f"Keypoint extraction completed. Cache size: {len(self.keypoint_cache)}")

    def calculate_keypoint_recall(self, human_keypoints: List[str], gpt4o_keypoints: List[str]) -> Dict:
        """Calculate recall between human and GPT4o key points with semantic matching"""
        try:
            if not human_keypoints or not gpt4o_keypoints:
                return {"recall": 0.0, "precision": 0.0, "f1": 0.0, "covered_points": []}
            
            # Log keypoint counts for debugging
            print(f"  Comparing {len(human_keypoints)} human keypoints vs {len(gpt4o_keypoints)} GPT4o keypoints")
            print(f"  Total comparisons needed: {len(human_keypoints) * len(gpt4o_keypoints)}")
            
            covered_points = []
            total_human_points = len(human_keypoints)
            comparison_count = 0
            cache_hits = 0
            
            for i, human_point in enumerate(human_keypoints):
                # Check if human key point is covered in GPT4o key points
                human_words = set(human_point.lower().split())
                
                best_coverage = 0.0
                best_match_method = "none"
                
                for gpt_point in gpt4o_keypoints:
                    comparison_count += 1
                    
                    # Check semantic cache first (most expensive operation)
                    semantic_key = (human_point, gpt_point)
                    if semantic_key in self.semantic_cache:
                        cache_hits += 1
                        semantic_similarity = self.semantic_cache[semantic_key]
                        if semantic_similarity > best_coverage:
                            best_coverage = semantic_similarity
                            best_match_method = "semantic"
                        continue  # Skip other methods if we have cached semantic similarity
                    
                    gpt_words = set(gpt_point.lower().split())
                    
                    # Strategy 1: Original word overlap (for exact matches)
                    intersection = human_words.intersection(gpt_words)
                    union = human_words.union(gpt_words)
                    
                    if union:
                        jaccard = len(intersection) / len(union)
                        coverage_ratio = len(intersection) / len(human_words) if len(human_words) > 0 else 0
                        word_overlap_score = 0.6 * jaccard + 0.4 * coverage_ratio
                        
                        if word_overlap_score > best_coverage:
                            best_coverage = word_overlap_score
                            best_match_method = "word_overlap"
                    
                    # Strategy 2: Semantic similarity (for paraphrases) - only if not cached
                    try:
                        semantic_similarity = self.calculate_semantic_similarity(human_point, gpt_point)
                        if semantic_similarity > best_coverage:
                            best_coverage = semantic_similarity
                            best_match_method = "semantic"
                    except Exception as e:
                        print(f"Error calculating semantic similarity: {e}")
                    
                    # Strategy 3: GPT rephrasing validation (for complex cases)
                    if best_coverage < 0.6:  # Only use GPT if other methods fail
                        try:
                            # Check rephrase cache
                            rephrase_key = (human_point, gpt_point)
                            if rephrase_key in self.rephrase_cache:
                                cache_hits += 1
                                is_rephrasing = self.rephrase_cache[rephrase_key]
                            else:
                                is_rephrasing = self.is_valid_rephrasing(human_point, gpt_point)
                            
                            if is_rephrasing and 0.5 > best_coverage:  # GPT validation overrides low scores
                                best_coverage = 0.7  # Set to high score if GPT confirms
                                best_match_method = "gpt_validation"
                        except Exception as e:
                            print(f"Error in GPT rephrasing validation: {e}")
                
                # Adaptive threshold based on match method
                threshold = 0.5  # Default threshold
                if best_match_method == "semantic":
                    threshold = 0.7  # Higher threshold for semantic similarity
                elif best_match_method == "gpt_validation":
                    threshold = 0.5  # Lower threshold for GPT-validated matches
                
                if best_coverage >= threshold:
                    covered_points.append({
                        "index": i + 1,
                        "human_point": human_point,
                        "best_match_method": best_match_method,
                        "best_score": best_coverage
                    })
            
            recall = len(covered_points) / total_human_points if total_human_points > 0 else 0.0
            precision = len(covered_points) / len(gpt4o_keypoints) if gpt4o_keypoints else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # Log cache efficiency
            cache_efficiency = cache_hits / comparison_count if comparison_count > 0 else 0.0
            print(f"  Cache efficiency: {cache_hits}/{comparison_count} ({cache_efficiency:.1%})")
            
            return {
                "recall": recall,
                "precision": precision,
                "f1": f1,
                "covered_points": covered_points,
                "total_human_points": total_human_points,
                "total_gpt4o_points": len(gpt4o_keypoints),
                "match_methods": [point["best_match_method"] for point in covered_points]
            }
        
        except Exception as e:
            print(f"Error calculating keypoint recall: {e}")
            return {"recall": 0.0, "precision": 0.0, "f1": 0.0, "covered_points": []}

    def calculate_semantic_similarity(self, point1: str, point2: str) -> float:
        """Calculate semantic similarity using Sentence-BERT with improved model"""
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Check if sentence model is available
            if self.sentence_model is None:
                print("Warning: SentenceTransformer not available, skipping semantic similarity")
                return 0.0
            
            # Cache disabled - calculate directly
            # key = (point1, point2)
            # if key in self.semantic_cache:
            #     return self.semantic_cache[key]
            
            # Encode sentences
            embeddings = self.sentence_model.encode([point1, point2])
            
            # Calculate cosine similarity
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            result = float(similarity)
            
            # Cache the result
            self.semantic_cache[key] = result
            return result
            
        except Exception as e:
            print(f"Error in semantic similarity calculation: {e}")
            return 0.0

    def is_valid_rephrasing(self, human_point: str, gpt_point: str) -> bool:
        """Use GPT to judge if GPT point is a valid rephrasing of human point"""
        try:
            # Check cache first
            key = (human_point, gpt_point)
            if key in self.rephrase_cache:
                return self.rephrase_cache[key]
            
            prompt = f"""Judge if the GPT point is a valid rephrasing of the human point.

Human point: {human_point}
GPT point: {gpt_point}

Consider:
- Do they express the same core concept?
- Are they semantically equivalent?
- Is the GPT point a valid way to express the human point?

Answer only: true or false"""

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a semantic equivalence judge. Answer only true or false."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0.1
            )
            
            answer = response.choices[0].message.content.strip().lower()
            result = "true" in answer
            
            # Cache the result
            self.rephrase_cache[key] = result
            return result
            
        except Exception as e:
            print(f"Error in GPT rephrasing validation: {e}")
            return False

    def g_eval(self, question: str, human_answer: str, gpt4o_answer: str) -> Dict:
        """Perform G-Eval (GPT Judge) evaluation"""
        try:
            prompt = f"""You are an expert evaluator. Your task is to evaluate the quality of a model-generated answer to an open-ended question by comparing it with a human-written reference answer.

You should rate the model answer on the following criteria:

1. **Correctness (1–5)**: Is the answer factually correct based on the question?
2. **Coverage (1–5)**: Does the answer include all the key points from the reference answer?
3. **Faithfulness (1–5)**: Is the answer grounded in facts, and free from hallucinated or unsupported claims?
4. **Clarity (1–5)**: Is the answer clearly written and easy to understand?
5. **Overall Equivalence**: Are the model and reference answers semantically equivalent? (true/false)

### Evaluation Steps:
1. Read the question and understand what it is asking.
2. Read the human reference answer. Identify the key points and factual content.
3. Read the model answer. Check whether it covers the same facts, without hallucination or misleading information.
4. Provide ratings (1–5) for the above criteria.
5. Decide whether the answers are semantically equivalent.
6. Write a short explanation of your decision.

---

Question: {question}

Reference Answer: {human_answer}

Model Answer: {gpt4o_answer}

Return format (JSON):
{{
  "correctness": 5,
  "coverage": 4,
  "faithfulness": 5,
  "clarity": 4,
  "equivalent": true,
  "explanation": "The model answer includes all key points with high factual accuracy and no hallucination. It elaborates slightly more but stays on topic."
}}"""

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert evaluator assessing answer quality objectively."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.1
            )
            
            content = response.choices[0].message.content.strip()
            
            # Try to parse JSON
            try:
                # Remove markdown code blocks if present
                if content.startswith('```json'):
                    content = content[7:]
                if content.endswith('```'):
                    content = content[:-3]
                
                result = json.loads(content)
                return result
            except json.JSONDecodeError:
                # Fallback: return default values
                return {
                    "correctness": 3,
                    "coverage": 3,
                    "faithfulness": 3,
                    "clarity": 3,
                    "equivalent": False,
                    "explanation": "Error parsing evaluation result"
                }
        
        except Exception as e:
            print(f"Error in G-Eval: {e}")
            return {
                "correctness": 3,
                "coverage": 3,
                "faithfulness": 3,
                "clarity": 3,
                "equivalent": False,
                "explanation": "Error in evaluation"
            }

    def calculate_bleu_score(self, reference: str, candidate: str) -> Dict[str, float]:
        """Calculate BLEU scores for multiple n-gram orders"""
        try:
            from sacrebleu import BLEU
            
            # sacrebleu expects lists of references and candidates
            bleu = BLEU()
            scores = {}
            
            # Calculate BLEU for each n-gram order
            for n in self.bleu_ngrams:
                try:
                    # For sacrebleu, we need to use the corpus_bleu method with specific n-gram weights
                    # Set weights to focus on the specific n-gram order
                    weights = [0] * 4  # Initialize with zeros
                    weights[n-1] = 1.0  # Set the specific n-gram weight to 1
                    
                    # Calculate BLEU with specific n-gram focus
                    score = bleu.sentence_score(candidate, [reference])
                    
                    # For n-gram specific scoring, we'll use nltk as fallback
                    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
                    reference_tokens = reference.split()
                    candidate_tokens = candidate.split()
                    
                    # Use smoothing function for short sentences
                    smooth = SmoothingFunction().method1
                    ngram_score = sentence_bleu([reference_tokens], candidate_tokens, 
                                              weights=weights, smoothing_function=smooth)
                    
                    scores[f'bleu_{n}'] = ngram_score
                    
                except Exception as e:
                    print(f"Error calculating BLEU-{n}: {e}")
                    scores[f'bleu_{n}'] = 0.0
            
            # Also calculate overall BLEU (equal weights for all n-grams)
            try:
                overall_score = bleu.sentence_score(candidate, [reference])
                scores['bleu'] = overall_score.score / 100.0  # Convert to 0-1 scale
            except Exception as e:
                print(f"Error calculating overall BLEU: {e}")
                scores['bleu'] = 0.0
            
            return scores
            
        except ImportError:
            # Fallback to nltk if sacrebleu not available
            try:
                from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
                reference_tokens = reference.split()
                candidate_tokens = candidate.split()
                smooth = SmoothingFunction().method1
                
                scores = {}
                
                # Calculate BLEU for each n-gram order
                for n in self.bleu_ngrams:
                    try:
                        weights = [0] * 4
                        weights[n-1] = 1.0
                        ngram_score = sentence_bleu([reference_tokens], candidate_tokens, 
                                                  weights=weights, smoothing_function=smooth)
                        scores[f'bleu_{n}'] = ngram_score
                    except Exception as e:
                        print(f"Error calculating BLEU-{n}: {e}")
                        scores[f'bleu_{n}'] = 0.0
                
                # Overall BLEU
                overall_score = sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smooth)
                scores['bleu'] = overall_score
                
                return scores
                
            except Exception as e:
                print(f"Error calculating BLEU score: {e}")
                return {f'bleu_{n}': 0.0 for n in self.bleu_ngrams}
        except Exception as e:
            print(f"Error calculating BLEU score: {e}")
            return {f'bleu_{n}': 0.0 for n in self.bleu_ngrams}

    def calculate_rouge_scores(self, reference: str, candidate: str) -> Dict:
        """Calculate ROUGE scores"""
        try:
            scores = self.rouge_scorer.score(reference, candidate)
            return {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            }
        except Exception as e:
            print(f"Error calculating ROUGE scores: {e}")
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

    def calculate_bertscore(self, references: List[str], candidates: List[str]) -> List[float]:
        """Calculate BERTScore"""
        try:
            P, R, F1 = bert_score(candidates, references, lang='en', device=self.device)
            return F1.tolist()
        except Exception as e:
            print(f"Error calculating BERTScore: {e}")
            return [0.0] * len(references)

    def evaluate_gpt4o_vs_human(self) -> Dict:
        """Evaluate GPT4o answers against human answers"""
        print("Starting GPT4o vs Human evaluation...")
        
        evaluation_results = []
        all_scores = {
            'bleu': [],
            'bleu_1': [],
            'bleu_2': [],
            'bleu_3': [],
            'bleu_4': [],
            'rouge1': [],
            'rouge2': [],
            'rougeL': [],
            'bertscore': [],
            'length_ratio': [],
            'keypoint_recall': [],
            'keypoint_precision': [],
            'keypoint_f1': [],
            'g_eval_correctness': [],
            'g_eval_coverage': [],
            'g_eval_faithfulness': [],
            'g_eval_clarity': [],
            'g_eval_equivalent': [],
            'match_methods': {
                'word_overlap': 0,
                'semantic': 0,
                'gpt_validation': 0,
                'none': 0
            }
        }
        
        for i, item in enumerate(self.data):
            print(f"Processing item {i+1}/{len(self.data)}...")
            
            human_answer = item.get('human_answer', '')
            gpt4o_answer = item.get('gpt4o_answer', '')
            question = item.get('question', '')
            
            if not human_answer or not gpt4o_answer or not question:
                continue
            
            # Extract key points from both answers
            human_keypoints = self.extract_key_points(question, human_answer)
            gpt4o_keypoints = self.extract_key_points(question, gpt4o_answer)
            
            # Calculate keypoint recall
            keypoint_result = self.calculate_keypoint_recall(human_keypoints, gpt4o_keypoints)
            
            # Calculate traditional metrics
            bleu_scores = self.calculate_bleu_score(human_answer, gpt4o_answer)
            rouge_scores = self.calculate_rouge_scores(human_answer, gpt4o_answer)
            bert_scores = self.calculate_bertscore([human_answer], [gpt4o_answer])
            bertscore = bert_scores[0] if bert_scores else 0.0
            length_ratio = min(len(gpt4o_answer), len(human_answer)) / max(len(gpt4o_answer), len(human_answer))
            
            # Perform G-Eval
            g_eval_result = self.g_eval(question, human_answer, gpt4o_answer)
            
            # Store result
            result = {
                'id': item['id'],
                'question': question,
                'human_answer': human_answer,
                'gpt4o_answer': gpt4o_answer,
                'human_keypoints': human_keypoints,
                'gpt4o_keypoints': gpt4o_keypoints,
                'scores': {
                    'bleu': bleu_scores,
                    'rouge': rouge_scores,
                    'bertscore': bertscore,
                    'length_ratio': length_ratio,
                    'keypoint_recall': keypoint_result,
                    'g_eval': g_eval_result
                }
            }
            
            evaluation_results.append(result)
            
            # Collect scores for aggregation
            for metric, score in bleu_scores.items():
                all_scores[metric].append(score)
            all_scores['rouge1'].append(rouge_scores['rouge1'])
            all_scores['rouge2'].append(rouge_scores['rouge2'])
            all_scores['rougeL'].append(rouge_scores['rougeL'])
            all_scores['bertscore'].append(bertscore)
            all_scores['length_ratio'].append(length_ratio)
            all_scores['keypoint_recall'].append(keypoint_result['recall'])
            all_scores['keypoint_precision'].append(keypoint_result['precision'])
            all_scores['keypoint_f1'].append(keypoint_result['f1'])
            
            # Collect match method statistics
            for method in keypoint_result.get('match_methods', []):
                if method in all_scores['match_methods']:
                    all_scores['match_methods'][method] += 1
            
            all_scores['g_eval_correctness'].append(g_eval_result['correctness'])
            all_scores['g_eval_coverage'].append(g_eval_result['coverage'])
            all_scores['g_eval_faithfulness'].append(g_eval_result['faithfulness'])
            all_scores['g_eval_clarity'].append(g_eval_result['clarity'])
            all_scores['g_eval_equivalent'].append(1 if g_eval_result['equivalent'] else 0)
            
            # Add delay to avoid rate limiting
            time.sleep(0.5)
        
        # Calculate aggregate statistics
        aggregate_stats = {}
        for metric, scores in all_scores.items():
            if metric != 'match_methods' and scores:
                aggregate_stats[metric] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'min': np.min(scores),
                    'max': np.max(scores),
                    'median': np.median(scores)
                }
        
        # Analyze match methods using Counter
        all_match_methods = []
        for item in evaluation_results:
            match_methods = item['scores']['keypoint_recall'].get('match_methods', [])
            all_match_methods.extend(match_methods)
        
        match_method_counts = Counter(all_match_methods)
        aggregate_stats['match_methods'] = dict(match_method_counts)
        
        # Calculate cache efficiency statistics
        total_comparisons = 0
        total_cache_hits = 0
        
        for item in evaluation_results:
            keypoint_result = item['scores']['keypoint_recall']
            total_comparisons += keypoint_result.get('comparison_count', 0)
            total_cache_hits += keypoint_result.get('cache_hits', 0)
        
        cache_stats = {
            'rephrase_cache_entries': len(self.rephrase_cache),
            'semantic_cache_entries': len(self.semantic_cache),
            'keypoint_cache_entries': len(self.keypoint_cache),
            'total_comparisons': total_comparisons,
            'total_cache_hits': total_cache_hits,
            'overall_cache_efficiency': total_cache_hits / total_comparisons if total_comparisons > 0 else 0.0,
            'average_comparisons_per_item': total_comparisons / len(evaluation_results) if evaluation_results else 0.0,
            'keypoint_cache_coverage': len(self.keypoint_cache) / (len(evaluation_results) * 2) if evaluation_results else 0.0  # 2 keypoints per item (human + gpt4o)
        }
        aggregate_stats['cache_efficiency'] = cache_stats
        
        return {
            'evaluation_results': evaluation_results,
            'aggregate_stats': aggregate_stats,
            'total_evaluated': len(evaluation_results)
        }

    def save_cache(self, cache_file: str = "evaluation_cache.json"):
        """Save cache to file for reuse"""
        def convert_cache_for_json(cache_dict):
            """Convert cache dictionary to JSON-serializable format"""
            json_cache = {}
            for key, value in cache_dict.items():
                if isinstance(key, tuple):
                    # Convert tuple keys to string representation
                    json_key = f"{key[0]}|{key[1]}"
                else:
                    json_key = str(key)
                json_cache[json_key] = value
            return json_cache
        
        cache_data = {
            'rephrase_cache': convert_cache_for_json(self.rephrase_cache),
            'semantic_cache': convert_cache_for_json(self.semantic_cache),
            'keypoint_cache': self.keypoint_cache  # This one already has string keys
        }
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
        print(f"Cache saved to: {cache_file}")

    def load_cache(self, cache_file: str = "evaluation_cache.json"):
        """Load cache from file"""
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            def convert_json_to_cache(json_cache):
                """Convert JSON cache back to original format with tuple keys"""
                cache = {}
                for key, value in json_cache.items():
                    if '|' in key:
                        # Convert back from string representation to tuple
                        parts = key.split('|', 1)
                        cache[(parts[0], parts[1])] = value
                    else:
                        cache[key] = value
                return cache
            
            self.rephrase_cache = convert_json_to_cache(cache_data.get('rephrase_cache', {}))
            self.semantic_cache = convert_json_to_cache(cache_data.get('semantic_cache', {}))
            self.keypoint_cache = cache_data.get('keypoint_cache', {})
            
            print(f"Cache loaded from: {cache_file}")
            print(f"  Rephrase cache entries: {len(self.rephrase_cache)}")
            print(f"  Semantic cache entries: {len(self.semantic_cache)}")
            print(f"  Keypoint cache entries: {len(self.keypoint_cache)}")
        else:
            print(f"Cache file not found: {cache_file}")

    def save_evaluation_results(self, results: Dict, output_file: str):
        """Save evaluation results"""
        # Convert numpy types to Python native types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # Convert all numpy types in results
        results_converted = convert_numpy_types(results)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_converted, f, indent=2, ensure_ascii=False)
        print(f"Evaluation results saved to: {output_file}")

    def print_evaluation_summary(self, results: Dict):
        """Print evaluation summary"""
        print("\n" + "="*60)
        print("GPT4o vs Human Answers Evaluation Summary")
        print("="*60)
        print(f"Total answers evaluated: {results['total_evaluated']}")
        
        print(f"\nAggregate Metrics:")
        for metric, stats in results['aggregate_stats'].items():
            if metric != 'match_methods' and metric != 'cache_efficiency':  # Handle special cases separately
                print(f"\n{metric.replace('_', ' ').title()}:")
                print(f"  Mean: {stats['mean']:.4f}")
                print(f"  Std:  {stats['std']:.4f}")
                print(f"  Range: {stats['min']:.4f} - {stats['max']:.4f}")
                print(f"  Median: {stats['median']:.4f}")
        
        # Print BLEU scores in a grouped format
        bleu_metrics = [k for k in results['aggregate_stats'].keys() if k.startswith('bleu')]
        if bleu_metrics:
            print(f"\nBLEU Scores:")
            for metric in sorted(bleu_metrics):
                stats = results['aggregate_stats'][metric]
                print(f"  {metric.upper()}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        # Print match method statistics
        if 'match_methods' in results['aggregate_stats']:
            print(f"\nKeypoint Match Methods Analysis:")
            match_counts = results['aggregate_stats']['match_methods']
            total_matches = sum(match_counts.values())
            
            if total_matches > 0:
                print(f"Total keypoint matches: {total_matches}")
                for method, count in match_counts.items():
                    percentage = (count / total_matches * 100)
                    print(f"  {method.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
                
                # Analysis insights
                print(f"\nMatch Method Insights:")
                if match_counts.get('semantic', 0) > match_counts.get('word_overlap', 0):
                    print("  ✅ Semantic similarity is the dominant matching method")
                    print("  → Indicates significant expression differences between human and GPT keypoints")
                else:
                    print("  ✅ Word overlap is the dominant matching method")
                    print("  → Indicates similar expression patterns between human and GPT keypoints")
                
                if match_counts.get('gpt_validation', 0) > 0:
                    print(f"  ✅ GPT validation resolved {match_counts['gpt_validation']} complex cases")
                    print("  → Helped with ambiguous semantic relationships")
            else:
                print("  No keypoint matches found")
        
        # Print cache efficiency
        if 'cache_efficiency' in results['aggregate_stats']:
            cache_stats = results['aggregate_stats']['cache_efficiency']
            print(f"\nCache Efficiency:")
            print(f"  Rephrase cache entries: {cache_stats['rephrase_cache_entries']}")
            print(f"  Semantic cache entries: {cache_stats['semantic_cache_entries']}")
            print(f"  Keypoint cache entries: {cache_stats['keypoint_cache_entries']}")
            print(f"  Keypoint cache coverage: {cache_stats['keypoint_cache_coverage']:.1%}")
            print(f"  Total comparisons: {cache_stats['total_comparisons']}")
            print(f"  Total cache hits: {cache_stats['total_cache_hits']}")
            print(f"  Overall cache efficiency: {cache_stats['overall_cache_efficiency']:.1%}")
            print(f"  Average comparisons per item: {cache_stats['average_comparisons_per_item']:.2f}")
            print(f"  → Caching saved redundant API calls and model computations")
        
        # Conclusion
        mean_bertscore = results['aggregate_stats']['bertscore']['mean']
        mean_keypoint_recall = results['aggregate_stats']['keypoint_recall']['mean']
        mean_g_eval_coverage = results['aggregate_stats']['g_eval_coverage']['mean']
        mean_g_eval_equivalent = results['aggregate_stats']['g_eval_equivalent']['mean']
        
        print(f"\nConclusion:")
        print(f"BERTScore (semantic similarity): {mean_bertscore:.3f}")
        print(f"Keypoint Recall: {mean_keypoint_recall:.3f}")
        print(f"G-Eval Coverage: {mean_g_eval_coverage:.3f}")
        print(f"G-Eval Equivalence: {mean_g_eval_equivalent:.1%}")
        
        if mean_bertscore > 0.8 and mean_keypoint_recall > 0.7 and mean_g_eval_coverage > 4.0 and mean_g_eval_equivalent > 0.7:
            print("✅ GPT4o shows strong equivalence to human answers - suitable replacement")
        elif mean_bertscore > 0.7 and mean_keypoint_recall > 0.5 and mean_g_eval_coverage > 3.5 and mean_g_eval_equivalent > 0.5:
            print("⚠️  GPT4o shows moderate equivalence - may be acceptable with limitations")
        else:
            print("❌ GPT4o shows low equivalence to human answers - not suitable replacement")
        
        print(f"\nNote: Keypoint recall and G-Eval provide more nuanced assessment than traditional metrics.")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Evaluate GPT4o answers vs human answers')
    parser.add_argument('--pre-extract', action='store_true', 
                       help='Pre-extract all keypoints before evaluation')
    parser.add_argument('--skip-evaluation', action='store_true',
                       help='Only pre-extract keypoints, skip evaluation')
    parser.add_argument('--bleu-ngrams', type=int, nargs='+', default=[1, 2, 3, 4],
                       help='BLEU n-gram orders to calculate (default: 1 2 3 4)')
    args = parser.parse_args()
    
    # Configuration
    gpt4o_answers_file = "gpt4o_answers/gpt4o_answers_dataset.json"
    evaluation_output = "gpt4o_vs_human_evaluation.json"
    cache_file = "evaluation_cache.json"
    
    # Check if GPT4o answers exist
    if not os.path.exists(gpt4o_answers_file):
        print(f"Error: {gpt4o_answers_file} not found")
        print("Please run generate_gpt4o_answers.py first")
        return
    
    # Initialize evaluator with BLEU n-gram configuration
    evaluator = GPT4oEvaluator(gpt4o_answers_file, bleu_ngrams=args.bleu_ngrams)
    
    # Load existing cache if available
    evaluator.load_cache(cache_file)
    
    # Pre-extract keypoints if requested
    if args.pre_extract or args.skip_evaluation:
        evaluator.pre_extract_all_keypoints()
        evaluator.save_cache(cache_file)
        
        if args.skip_evaluation:
            print("Keypoint pre-extraction completed. Exiting.")
            return
    
    # Evaluate
    results = evaluator.evaluate_gpt4o_vs_human()
    
    # Save results
    evaluator.save_evaluation_results(results, evaluation_output)
    
    # Save cache for future use
    evaluator.save_cache(cache_file)
    
    # Print summary
    evaluator.print_evaluation_summary(results)

if __name__ == "__main__":
    main() 