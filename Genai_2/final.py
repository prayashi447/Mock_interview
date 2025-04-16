import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuration
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DATA_PATH = DATA_DIR / "raw/questions.csv"
EMBEDDING_MODEL = 'BAAI/bge-large-en-v1.5'
LLM_MODEL = 'microsoft/phi-2'

# Load the embedding model
encoder = SentenceTransformer(EMBEDDING_MODEL)

class InterviewQuestionGenerator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.load_data()
        self.load_models()
        self.build_index()
        
    def load_data(self):
        """Load the embeddings and metadata"""
        print("Loading embeddings and metadata...")
        self.question_embeddings = np.load(PROCESSED_DIR / "questions.npy")
        self.answer_embeddings = np.load(PROCESSED_DIR / "answers.npy")
        self.category_embeddings = np.load(PROCESSED_DIR / "categories.npy")
        
        with open(PROCESSED_DIR / "metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
            self.metadata = metadata['records']
            self.category_mapping = metadata['category_mapping']
            self.category_embeddings_meta = metadata['category_embeddings']
            
        self.df = pd.read_csv(RAW_DATA_PATH)
        
    def load_models(self):
        """Load the language model"""
        print(f"Loading language model {LLM_MODEL}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            LLM_MODEL, 
            trust_remote_code=True,
            padding_side="left"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.llm = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            trust_remote_code=True
        ).to(self.device)
        
    def build_index(self):
        """Build nearest neighbors index for questions and categories"""
        print("Building search indexes...")
        self.question_index = NearestNeighbors(n_neighbors=5, metric='cosine')
        self.question_index.fit(self.question_embeddings)
        
        self.category_index = NearestNeighbors(n_neighbors=1, metric='cosine')
        self.category_index.fit(self.category_embeddings_meta)
        
    def get_category(self, query): #This line encodes the input query into an embedding vector using the SentenceTransformer(EMBEDDING_MODEL) model.
        """Find the most relevant category for a query"""
        query_embedding = encoder.encode(query, normalize_embeddings=True)
        _, indices = self.category_index.kneighbors([query_embedding])
        category_idx = indices[0][0]
        return self.category_mapping[category_idx] #closest category to the query's embedding
        
    def retrieve_questions(self, query, category=None, n=3): #compared against precomputed question embeddings.
        """Retrieve similar questions based on the query"""
        query_embedding = encoder.encode(query, normalize_embeddings=True)
        
        if category:
            # Filter questions by category first
            cat_questions = [i for i, m in enumerate(self.metadata) 
                           if m['Category'] == category]
            filtered_embeddings = self.question_embeddings[cat_questions]
            
            # Build temporary index for this category
            temp_index = NearestNeighbors(n_neighbors=min(n, len(cat_questions)), 
                                        metric='cosine')
            temp_index.fit(filtered_embeddings)
            
            _, indices = temp_index.kneighbors([query_embedding])
            original_indices = [cat_questions[i] for i in indices[0]]
        else:
            # Search across all questions
            _, indices = self.question_index.kneighbors([query_embedding])
            original_indices = indices[0][:n]
            
        return [self.df.iloc[i] for i in original_indices]
    
    def generate_question(self, context, category, temperature=0.7, max_new_tokens=100):
        """Generate a new question using Phi-2"""
        prompt = f"""You are a technical interviewer. Based on the context below, generate a new technical interview question about {category}.
        
Context:
{context}

New technical interview question:"""
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            return_attention_mask=True,
            truncation=True,
            max_length=2048  # Phi-2's context window
        ).to(self.device)
        
        outputs = self.llm.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        generated_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[-1]:], 
            skip_special_tokens=True
        )
        return generated_text.strip()
    
    def rag_pipeline(self, topic, category=None, n_retrieve=3, n_generate=2):
        """
        Full RAG pipeline:
        1. Determine category if not provided
        2. Retrieve relevant questions
        3. Generate new questions based on retrieved context
        """
        if not category:
            category = self.get_category(topic)
            print(f"Automatically determined category: {category}")
            
        # Retrieve similar questions
        retrieved = self.retrieve_questions(topic, category, n=n_retrieve)
        print("\nRetrieved similar questions:")
        for i, q in enumerate(retrieved, 1):
            print(f"{i}. {q['Cleaned_Question']}")
            print(f"   Answer: {q['Cleaned_Answer'][:100]}...\n")
        
        # Generate context from retrieved questions
        context = "\n".join([
            f"Question: {q['Cleaned_Question']}\nAnswer: {q['Cleaned_Answer']}" 
            for q in retrieved
        ])
        
        # Generate new questions
        print("\nGenerated questions:")
        generated = []
        for _ in range(n_generate):
            new_question = self.generate_question(context, category)
            generated.append(new_question)
            print(f"- {new_question}")
            
        return {
            "category": category,
            "retrieved_questions": retrieved,
            "generated_questions": generated
        }

# Example usage
if __name__ == "__main__":
    generator = InterviewQuestionGenerator()
    
    # Example 1: With explicit category
    print("\n" + "="*50)
    print("Example 1: SQL Question Generation")
    print("="*50)
    generator.rag_pipeline(
        topic="database optimization techniques",
        category="SQL",
        n_retrieve=2,
        n_generate=2
    )
    
    # Example 2: Let the system determine category
    print("\n" + "="*50)
    print("Example 2: AI/Data Science Question Generation")
    print("="*50)
    generator.rag_pipeline(
        topic="neural network architectures for image recognition",
        n_retrieve=3,
        n_generate=2
    )