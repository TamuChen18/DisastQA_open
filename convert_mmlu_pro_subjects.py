#!/usr/bin/env python3
"""
Convert selected MMLU-PRO subjects into the same MCQ format used in DisastQA.
Target subjects: Medicine, Biology, Geography, Global Facts, Law, Sociology, Public Relations, Psychology, Economics
"""
import json
import os
import subprocess
import sys

# Target subjects (8 subjects × 250 questions = 2000 total)
TARGET_SUBJECTS = [
    "health", "biology", "law", "psychology", "economics", 
    "business", "chemistry", "engineering"
]

def install_pandas():
    """Try to install pandas and pyarrow if not available."""
    try:
        import pandas as pd
        import pyarrow
        return True
    except ImportError:
        print("pandas and pyarrow are required to process parquet files.")
        print("Attempting to install...")
        
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "--break-system-packages", "pandas", "pyarrow"])
            print("pandas and pyarrow installed successfully.")
            return True
        except subprocess.CalledProcessError:
            print("Installation failed. Please install manually:")
            print("pip install pandas pyarrow")
            return False

def convert_mmlu_pro_subjects_to_mcq_format():
    """Convert selected MMLU-PRO subjects into DisastQA-style MCQ format."""
    
    # Input and output paths
    parquet_file = "DATA/MMLUE-PRO/data/test-00000-of-00001.parquet"
    output_file = "DATA/MMLUE-PRO/mmlu_pro_subjects_mcq_format.json"
    
    if not os.path.exists(parquet_file):
        print(f"File not found: {parquet_file}")
        return False
    
    print("Reading MMLU-PRO dataset...")
    print(f"Target subjects: {', '.join(TARGET_SUBJECTS)}")
    
    # Ensure pandas is available
    if not install_pandas():
        print("Unable to install pandas. Creating sample data instead...")
        create_sample_subjects_data(output_file)
        return True
    
    try:
        import pandas as pd
        
        # Load parquet file
        df = pd.read_parquet(parquet_file)
        print(f"Loaded successfully: {len(df)} samples.")
        
        # Display data structure
        print("\nColumns:", list(df.columns))
        
        # Check subject distribution
        if 'category' in df.columns:
            print("\nAll subject distribution:")
            category_counts = df['category'].value_counts()
            for category, count in category_counts.items():
                print(f"  {category}: {count}")
            
            # Filter for target subjects
            print("\nFiltering target subjects...")
            target_df = df[df['category'].isin(TARGET_SUBJECTS)]
            print(f"Filtered samples: {len(target_df)}")
            
            print("\nFiltered subject distribution:")
            target_category_counts = target_df['category'].value_counts()
            for category, count in target_category_counts.items():
                print(f"  {category}: {count}")
            
            # Sample 250 per subject
            print("\nSampling 250 per subject...")
            sampled_data = []
            for category in TARGET_SUBJECTS:
                category_data = target_df[target_df['category'] == category]
                if len(category_data) >= 250:
                    sampled_category = category_data.sample(n=250, random_state=42)
                else:
                    sampled_category = category_data
                    print(f"Warning: {category} only has {len(category_data)} samples. Using all.")
                
                sampled_data.append(sampled_category)
                print(f"  {category}: {len(sampled_category)} samples selected.")
            
            # Combine all sampled subjects
            target_df = pd.concat(sampled_data, ignore_index=True)
            print(f"\nFinal total samples: {len(target_df)}")
            
            print("\nFinal subject distribution:")
            final_counts = target_df['category'].value_counts()
            for category, count in final_counts.items():
                print(f"  {category}: {count}")
                
        else:
            print("Error: 'category' column not found in dataset.")
            return False
        
        # Convert to MCQ format
        print("\nConverting to MCQ format...")
        mcq_data = []
        
        for idx, row in target_df.iterrows():
            options = row['options'].tolist() if hasattr(row['options'], 'tolist') else list(row['options'])
            options_list = [f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)]
            
            answer_index = int(row['answer_index'])
            if 0 <= answer_index < len(options):
                correct_option_char = chr(65 + answer_index)
            else:
                correct_option_char = "UNKNOWN"
                print(f"Warning: Question ID {row.get('question_id', 'N/A')} has invalid answer index {answer_index} (options: {len(options)})")
            
            mcq_item = {
                "passage": "",
                "category": str(row['category']),
                "multiple_choice": {
                    "gpt40": {
                        "content": {
                            "question": str(row['question']),
                            "options": options_list,
                            "correct_option": correct_option_char
                        }
                    }
                }
            }
            mcq_data.append(mcq_item)
        
        # Save output
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(mcq_data, f, indent=2, ensure_ascii=False)
        
        print(f"Conversion completed: {len(mcq_data)} samples saved to {output_file}")
        
        # Preview sample
        print("\nConverted data structure:")
        print("Keys:", list(mcq_data[0].keys()))
        sample = mcq_data[0]
        print(f"category: {sample['category']}")
        print(f"passage: '{sample['passage']}'")
        print(f"question: {sample['multiple_choice']['gpt40']['content']['question'][:100]}...")
        print(f"options: {sample['multiple_choice']['gpt40']['content']['options']}")
        print(f"correct_option: {sample['multiple_choice']['gpt40']['content']['correct_option']}")
        
        # Summary statistics
        print("\nSummary statistics:")
        print(f"Total questions: {len(mcq_data)}")
        
        subject_counts = {}
        for item in mcq_data:
            subject = item['category']
            subject_counts[subject] = subject_counts.get(subject, 0) + 1
        
        print("\nQuestions per subject:")
        for subject, count in subject_counts.items():
            print(f"  {subject}: {count}")
        
        option_counts = {}
        for item in mcq_data:
            num_options = len(item['multiple_choice']['gpt40']['content']['options'])
            option_counts[num_options] = option_counts.get(num_options, 0) + 1
        
        print("\nOption count distribution:")
        for num_options, count in sorted(option_counts.items()):
            print(f"  {num_options} options: {count} questions")
            
        return True
            
    except Exception as e:
        print(f"Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_sample_subjects_data(output_file):
    """Create sample data for demonstration (when pandas is unavailable)."""
    print("Creating sample data for target subjects...")
    
    sample_data = []
    
    sample_questions = {
        "Medicine": {
            "question": "What is the most common cause of acute myocardial infarction?",
            "options": ["Coronary artery spasm", "Atherosclerosis", "Viral infection", "Trauma", "Drug toxicity", "Genetic mutation", "Autoimmune disease", "Metabolic disorder", "Infectious disease", "Environmental factors"],
            "correct_option": "B"
        },
        "Biology": {
            "question": "Which organelle is responsible for protein synthesis?",
            "options": ["Mitochondria", "Ribosome", "Nucleus", "Golgi apparatus", "Endoplasmic reticulum", "Lysosome", "Chloroplast", "Vacuole", "Cell membrane", "Cytoplasm"],
            "correct_option": "B"
        },
        "Geography": {
            "question": "What is the longest river in the world?",
            "options": ["Amazon", "Nile", "Mississippi", "Yangtze", "Yellow", "Ganges", "Danube", "Rhine", "Thames", "Seine"],
            "correct_option": "B"
        },
        "Global Facts": {
            "question": "Which country has the largest population?",
            "options": ["India", "China", "United States", "Indonesia", "Pakistan", "Brazil", "Nigeria", "Bangladesh", "Russia", "Mexico"],
            "correct_option": "B"
        },
        "Law": {
            "question": "What is the principle of 'innocent until proven guilty' called?",
            "options": ["Due process", "Presumption of innocence", "Burden of proof", "Habeas corpus", "Double jeopardy", "Miranda rights", "Statute of limitations", "Res judicata", "Stare decisis", "Actus reus"],
            "correct_option": "B"
        },
        "Sociology": {
            "question": "What term describes the process by which individuals learn the norms and values of their society?",
            "options": ["Assimilation", "Socialization", "Acculturation", "Integration", "Enculturation", "Indoctrination", "Education", "Training", "Conditioning", "Programming"],
            "correct_option": "B"
        },
        "Public Relations": {
            "question": "What is the primary goal of crisis communication?",
            "options": ["Increase sales", "Protect reputation", "Entertain audience", "Educate public", "Promote products", "Generate leads", "Build awareness", "Create controversy", "Avoid responsibility", "Maximize profit"],
            "correct_option": "B"
        },
        "Psychology": {
            "question": "Which psychological theory emphasizes the role of unconscious processes?",
            "options": ["Behaviorism", "Psychoanalysis", "Cognitive psychology", "Humanistic psychology", "Social psychology", "Developmental psychology", "Biological psychology", "Evolutionary psychology", "Positive psychology", "Gestalt psychology"],
            "correct_option": "B"
        },
        "Economics": {
            "question": "What is the economic term for the total value of all goods and services produced in a country?",
            "options": ["Net National Product", "Gross Domestic Product", "Gross National Product", "Net Domestic Product", "National Income", "Personal Income", "Disposable Income", "Real Income", "Nominal Income", "Per Capita Income"],
            "correct_option": "B"
        }
    }
    
    for subject, question_data in sample_questions.items():
        sample_item = {
            "passage": "",
            "category": subject,
            "multiple_choice": {
                "gpt40": {
                    "content": {
                        "question": question_data["question"],
                        "options": question_data["options"],
                        "correct_option": question_data["correct_option"]
                    }
                }
            }
        }
        sample_data.append(sample_item)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    print(f"Sample data created for {len(sample_data)} subjects.")
    print(f"Saved to: {output_file}")
    print("\nNote: This is only demonstration data. For full processing, pandas must be installed.")

if __name__ == "__main__":
    convert_mmlu_pro_subjects_to_mcq_format()
