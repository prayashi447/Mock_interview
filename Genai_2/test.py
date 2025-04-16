import numpy as np
import spacy
import pickle
import re
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pandas as pd
from pathlib import Path

# Configuration
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
RESUME_PATH = "resume.txt"  # Your input resume text file

def load_data():
    """Load all required data files with error handling"""
    try:
        questions_emb = np.load(PROCESSED_DIR / "questions.npy")
        answers_emb = np.load(PROCESSED_DIR / "answers.npy")
        categories_emb = np.load(PROCESSED_DIR / "categories.npy")
        
        with open(PROCESSED_DIR / "metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
            
        # Validate metadata format
        if not all(isinstance(m, dict) for m in metadata):
            print("Warning: Converting metadata to proper format")
            metadata = [{"ID": i, "Category": str(m), "Difficulty": "Medium"} 
                       if not isinstance(m, dict) else m 
                       for i, m in enumerate(metadata)]
            
        return questions_emb, answers_emb, categories_emb, metadata
    
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

# Load preprocessed data
questions_emb, answers_emb, categories_emb, metadata = load_data()

# Load models
try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    raise ImportError("English language model not found. Run: python -m spacy download en_core_web_lg")

encoder = SentenceTransformer('BAAI/bge-large-en-v1.5')

# Enhanced Skill database with patterns
SKILLS_DB = {
    "AI (Data Science)": [
        "machine learning", "deep learning", "neural network", "neural networks",
        "pytorch", "tensorflow", "scikit.?learn", "data science",
        "natural language processing", "nlp", "computer vision", "bert", "transformer"
    ],
    "Software Engineering": [
        "python", "java", "c\\+\\+", "git", "docker", "kubernetes",
        "software development", "backend", "frontend", "database", "sql",
        "object.?oriented", "oop", "rest api", "microservices"
    ],
    "Cloud Computing": [
        "aws", "amazon web services", "azure", "gcp", "google cloud", "cloud",
        "serverless", "lambda", "ec2", "s3", "kubernetes", "terraform",
        "container", "iaas", "paas", "saas"
    ]
}

class ResumeProcessor:
    def __init__(self):
        self.nlp = nlp
        self.skills_db = SKILLS_DB
        
    def clean_text(self, text):
        """Clean resume text for better processing"""
        # First keep only alphanumeric, spaces, and + (for C++)
        text = re.sub(r'[^\w\s+]', ' ', text.lower())
        # Then collapse multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text
        
    def load_resume(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return self.clean_text(f.read())
        
    def extract_skills(self, text):
        """Improved skill extraction with regex patterns"""
        skills = set()
        text = self.load_resume(text) if isinstance(text, str) and len(text) < 256 else text
        
        # Match skills using regex patterns
        for category, patterns in self.skills_db.items():
            for pattern in patterns:
                try:
                    matches = re.finditer(r'(?<!\w)' + pattern + r'(?!\w)', text, re.IGNORECASE)
                    for match in matches:
                        skills.add((match.group(), category))
                except re.error:
                    continue
        
        return sorted(skills, key=lambda x: len(x[0]), reverse=True)  # Longer matches first
    
    def match_category(self, skills):
        if not skills:
            return None, 0.0
        
        # Count category occurrences
        category_counts = {}
        for _, category in skills:
            category_counts[category] = category_counts.get(category, 0) + 1
        
        if not category_counts:
            return None, 0.0
            
        # Get category with max count
        main_category = max(category_counts.keys(), key=lambda k: category_counts[k])
        confidence = category_counts[main_category] / len(skills)
        
        return main_category, round(confidence, 2)

class QuestionRetriever:
    def __init__(self):
        self.questions_emb = questions_emb
        self.metadata = metadata
        self.categories_emb = categories_emb
        self.category_names = list(SKILLS_DB.keys())
        
    def get_category_embedding(self, category_name):
        try:
            idx = self.category_names.index(category_name)
            return self.categories_emb[idx]
        except ValueError:
            return None
    
    def retrieve_questions(self, category_name, top_k=5):
        """Safe question retrieval with validation"""
        if not isinstance(self.metadata, list) or not self.metadata:
            print("Error: Invalid metadata format")
            return []
            
        try:
            category_questions = []
            for i, meta in enumerate(self.metadata):
                if not isinstance(meta, dict):
                    continue
                if str(meta.get('Category', '')).lower() == category_name.lower():
                    category_questions.append(i)
            
            if not category_questions:
                print(f"No questions found for category: {category_name}")
                print(f"Available categories: {set(m.get('Category', '') for m in self.metadata if isinstance(m, dict))}")
                return []
            
            # Sort by difficulty (handle missing values)
            sorted_questions = sorted(
                category_questions,
                key=lambda x: str(self.metadata[x].get('Difficulty', '')).lower()
            )[:top_k]
            
            return sorted_questions
            
        except Exception as e:
            print(f"Error retrieving questions: {e}")
            return []

def main():
    try:
        # 1. Process resume
        processor = ResumeProcessor()
        resume_text = processor.load_resume(RESUME_PATH)
        skills = processor.extract_skills(resume_text)
        main_category, confidence = processor.match_category(skills)
        
        print("\nExtracted Skills:")
        for skill, category in skills:
            print(f"- {skill} ({category})")
        
        print(f"\nMain Category: {main_category} (Confidence: {confidence:.2f})")
        
        if not main_category:
            print("No relevant skills found in resume")
            return
        
        # 2. Retrieve questions
        retriever = QuestionRetriever()
        question_ids = retriever.retrieve_questions(main_category)
        
        if not question_ids:
            print("No questions available for the identified category")
            return
        
        # 3. Load original questions
        try:
            df = pd.read_csv(DATA_DIR / "raw/questions.csv")
        except Exception as e:
            print(f"Error loading questions CSV: {e}")
            return
        
        print("\nRecommended Questions:")
        for qid in question_ids:
            try:
                print(f"\nID: {qid}")
                print(f"Question: {df.iloc[qid]['Question']}")
                print(f"Category: {df.iloc[qid]['Category']}")
                print(f"Difficulty: {df.iloc[qid]['Difficulty']}")
            except IndexError:
                print(f"\nQuestion ID {qid} not found in dataset")
                continue

    except Exception as e:
        print(f"\nError in main execution: {e}")

if __name__ == "__main__":
    main()