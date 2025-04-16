import numpy as np
import spacy
import pickle
import re
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Configuration
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"

# Initialize encoder
encoder = SentenceTransformer('BAAI/bge-large-en-v1.5')

def load_data():
    """Load all data files with error handling"""
    try:
        questions_emb = np.load(PROCESSED_DIR / "questions.npy")
        answers_emb = np.load(PROCESSED_DIR / "answers.npy")
        categories_emb = np.load(PROCESSED_DIR / "categories.npy")
        
        with open(PROCESSED_DIR / "metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
            
        return questions_emb, answers_emb, categories_emb, metadata
    
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

# Skill database with normalized categories
SKILLS_DB = {
    "AI (Data Science)": [
        "machine learning", "deep learning", "neural networks",
        "pytorch", "tensorflow", "scikit.?learn", "data science",
        "natural language processing", "nlp", "computer vision"
    ],
    "General Software Engineering": [
        "python", "java", "c\\+\\+", "git", "docker", "kubernetes",
        "software development", "backend", "frontend", "database"
    ],
    "Containers and Cloud": [
        "aws", "azure", "gcp", "cloud", "serverless",
        "lambda", "ec2", "s3", "kubernetes", "terraform"
    ],
    "SQL":["sql","mysql"],
    "DevOps":[
        "webdev","React"
    ]
}

class ResumeProcessor:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_lg")
        except OSError:
            raise ImportError("English language model not found. Run: python -m spacy download en_core_web_lg")
        self.skills_db = SKILLS_DB
    
    def clean_text(self, text):
        """Normalize text for processing"""
        text = re.sub(r'[^\w\s+]', ' ', text.lower())
        return re.sub(r'\s+', ' ', text).strip()
    
    def process_resume(self, resume_input):
        """Handle both file paths and direct text input"""
        if isinstance(resume_input, str) and len(resume_input) < 256 and Path(resume_input).exists():
            with open(resume_input, 'r', encoding='utf-8') as f:
                text = self.clean_text(f.read())
        else:
            text = self.clean_text(resume_input)
        return text
    
    def extract_skills(self, text):
        """Extract skills with proper word boundaries"""
        text = self.process_resume(text)
        skills = set()
        
        for category, terms in self.skills_db.items():
            for term in terms:
                pattern = r'(?<!\w)' + re.escape(term) + r'(?!\w)'
                if re.search(pattern, text, re.IGNORECASE):
                    skills.add((term, category))
        
        return sorted(skills, key=lambda x: len(x[0]), reverse=True)
    
    def match_category(self, skills):
        """Enhanced category matching"""
        if not skills:
            return None, 0.0
        
        # Frequency-based matching
        freq_counts = {}
        for _, category in skills:
            freq_counts[category] = freq_counts.get(category, 0) + 1
        
        # Embedding-based verification
        skill_text = ' '.join([s[0] for s in skills])
        skill_emb = encoder.encode([skill_text])[0]
        similarities = cosine_similarity([skill_emb], categories_emb)[0]
        best_idx = np.argmax(similarities)
        
        # Cross-validate
        main_category = max(freq_counts.keys(), key=lambda k: freq_counts[k])
        if list(SKILLS_DB.keys())[best_idx] == main_category:
            confidence = (freq_counts[main_category]/len(skills)) * similarities[best_idx]
        else:
            confidence = freq_counts[main_category]/len(skills)
        
        return main_category, round(confidence, 2)

class QuestionRetriever:
    def __init__(self):
        self.questions_emb, _, self.categories_emb, self.metadata = load_data()
    
    def retrieve_questions(self, category_name, top_k=5):
        """Retrieve questions using hybrid approach"""
        try:
            # Get category index
            cat_idx = list(SKILLS_DB.keys()).index(category_name)
            cat_embedding = self.categories_emb[cat_idx]
            
            # Find questions in this category
            candidate_indices = [
                i for i, m in enumerate(self.metadata)
                if isinstance(m, dict) and m.get("Category", "").lower() == category_name.lower()
            ]
            
            if not candidate_indices:
                print(f"No metadata matches for {category_name}")
                return []
            
            # Sort by embedding similarity
            question_embs = self.questions_emb[candidate_indices]
            similarities = cosine_similarity(question_embs, [cat_embedding]).flatten()
            
            # Get top matches
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            return [candidate_indices[i] for i in top_indices]
        
        except ValueError:
            print(f"Unknown category: {category_name}")
            return []

def main(resume_input=None):
    try:
        # Load all data
        global categories_emb
        questions_emb, answers_emb, categories_emb, metadata = load_data()
        
        # Initialize components
        processor = ResumeProcessor()
        retriever = QuestionRetriever()
        
        # Get resume text (use default if not provided)
        resume_text = resume_input or """
            John Doe
            Software Engineer | Data Scientist
            john.doe@email.com | (123) 456-7890 | linkedin.com/in/johndoe

            SUMMARY:
            Machine Learning Engineer with 3+ years of experience in Python, TensorFlow, and cloud technologies. 
            Specialized in NLP and computer vision applications.

            TECHNICAL SKILLS:
            - Programming: Python, Java, C++
            - Machine Learning: PyTorch, TensorFlow, scikit-learn
            - Cloud: AWS (EC2, S3), Azure, GCP
            - Databases: MySQL, MongoDB
            - Tools: Git, Docker, Kubernetes
            """
        
        # Process resume
        skills = processor.extract_skills(resume_text)
        print("\nExtracted Skills:")
        for skill, category in skills:
            print(f"- {skill} ({category})")
        
        main_category, confidence = processor.match_category(skills)
        print(f"\nMain Category: {main_category} (Confidence: {confidence:.2f})")
        
        if not main_category:
            print("No relevant skills found")
            return
        
        # Retrieve questions
        question_ids = retriever.retrieve_questions(main_category)
        
        if not question_ids:
            print("\nNo qualified questions found. Trying broader search...")
            question_ids = [
                i for i, m in enumerate(metadata)
                if isinstance(m, dict) and m.get("Category", "").lower() == main_category.lower()
            ][:5]
        
        # Display results
        df = pd.read_csv(DATA_DIR / "raw/questions.csv")
        print("\nRecommended Questions:")
        for qid in question_ids:
            try:
                print(f"\nID: {qid}")
                print(f"Question: {df.iloc[qid]['Question']}")
                print(f"Category: {df.iloc[qid]['Category']}")
                print(f"Difficulty: {df.iloc[qid]['Difficulty']}")
            except (IndexError, KeyError) as e:
                print(f"\n[Error] Question ID {qid} - {str(e)}")

    except Exception as e:
        print(f"\nSystem error: {str(e)}")

if __name__ == "__main__":
    # Usage: Can pass resume text or file path
    # main()  # Uses default resume
    # main("path/to/resume.txt")  # From file
    main("""John Doe
Software Engineer | Data Scientist
john.doe@email.com | (123) 456-7890 | linkedin.com/in/johndoe

SUMMARY:
Machine Learning Engineer with 3+ years experience in Python, TensorFlow, and cloud technologies. 
Specialized in NLP and computer vision applications.

TECHNICAL SKILLS:
- Programming: Python, Java, C++
- Machine Learning: PyTorch, TensorFlow, scikit-learn
- Cloud: AWS (EC2, S3), Azure, GCP
- Databases: MySQL, MongoDB
- Tools: Git, Docker, Kubernetes""")