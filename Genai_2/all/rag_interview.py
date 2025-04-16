import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
import random

# 1. Enhanced Data Loader with TF-IDF
def load_data():
    try:
        embeddings_dir = "interview_embeddings"
        data = {
            "questions": np.load(f"{embeddings_dir}/questions.npy"),
            "answers": np.load(f"{embeddings_dir}/answers.npy"),
            "model": SentenceTransformer('BAAI/bge-large-en-v1.5')
        }
        
        # Load metadata with fallback
        try:
            data["metadata"] = pd.read_parquet(f"{embeddings_dir}/metadata.parquet")
        except:
            data["metadata"] = pd.read_csv(f"{embeddings_dir}/metadata.csv")
            
        # Initialize TF-IDF vectorizer
        data["tfidf"] = TfidfVectorizer().fit(data["metadata"]['Cleaned_Question'])
        
        print(f"Loaded {len(data['metadata'])} questions across categories: {data['metadata']['Category'].unique()}")
        return data
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

# 2. Hybrid Retrieval (Semantic + Keyword)
def retrieve_questions(data, query: str, top_k: int = 3, min_score: float = 0.5):
    try:
        # Semantic similarity
        query_embed = data["model"].encode([query], normalize_embeddings=True)
        semantic_sim = np.dot(query_embed, data["questions"].T)[0]
        
        # Keyword relevance (TF-IDF)
        keyword_sim = data["tfidf"].transform([query]).toarray().mean(axis=1)
        
        # Combined score
        combined_scores = 0.7 * semantic_sim + 0.3 * keyword_sim
        
        # Get top matches
        top_indices = np.argsort(combined_scores)[-top_k:][::-1]
        results = data["metadata"].iloc[top_indices].copy()
        results['score'] = combined_scores[top_indices]
        
        # Filter by minimum score
        return results[results['score'] > min_score]
    
    except Exception as e:
        print(f"Retrieval error: {str(e)}")
        return pd.DataFrame()

# 3. Resume Generator (Dynamic based on dataset)
def generate_technical_resume(metadata):
    tech_terms = " ".join(metadata['Cleaned_Question']).split()
    key_terms = list(set([term for term in tech_terms if term not in ['what','is','the','a','an']]))
    
    resume = f"""
    TECHNICAL SKILLS:
    - {random.choice(['Expert','Proficient'])} in {random.choice(key_terms)} and {random.choice(key_terms)}
    - Experience with {random.choice(['implementing','optimizing'])} {random.choice(key_terms)}
    
    PROJECTS:
    - Developed {random.choice(['machine learning','data analysis'])} system using {random.choice(key_terms)}
    - Researched applications of {random.choice(key_terms)} in {random.choice(key_terms)}
    
    EDUCATION:
    - {random.choice(['BSc','MSc'])} in {random.choice(['Computer Science','AI'])}
    """
    return resume

# 4. Enhanced Evaluation with Phi-2
def evaluate_answer(question, answer, reference, tokenizer, model):
    prompt = f"""Evaluate this technical answer (1-10 scale):
    Question: {question}
    Answer: {answer}
    Reference Answer: {reference}
    
    Consider:
    1. Technical accuracy (40%)
    2. Explanation depth (30%)
    3. Clarity (20%)
    4. Examples provided (10%)
    
    Detailed Evaluation:"""
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(model.device)
    outputs = model.generate(**inputs, max_length=1000, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split("Detailed Evaluation:")[-1].strip()

# 5. Gradio Interface
def create_interface(data, phi2, tokenizer):
    def process_resume(resume_text):
        if not resume_text.strip():
            raise gr.Error("Please enter resume text")
            
        results = retrieve_questions(data, resume_text)
        if results.empty:
            sample_resume = generate_technical_resume(data["metadata"])
            raise gr.Error(f"No matches found. Try adding terms like: {sample_resume[:200]}...")
            
        return (
            results.iloc[0]['Question'],
            results.iloc[0]['Answer'],
            "\n".join(f"{row['Question']} (Score: {row['score']:.2f})" 
                    for _, row in results.iterrows())
        )

    with gr.Blocks(theme=gr.themes.Soft()) as app:
        gr.Markdown("## AI Technical Interview Simulator")
        
        with gr.Tab("Resume Analysis"):
            with gr.Row():
                resume_input = gr.Textbox(label="Paste Resume Text", lines=5, placeholder="e.g., Experience with Python and machine learning...")
                with gr.Column():
                    gr.Markdown("**Sample Resume**")
                    sample_resume = gr.Textbox(value=generate_technical_resume(data["metadata"]), interactive=False)
            submit_btn = gr.Button("Get Questions", variant="primary")
            
            question_output = gr.Textbox(label="Suggested Question", interactive=False)
            reference_output = gr.Textbox(label="Reference Answer", visible=False)
            matches_output = gr.Textbox(label="Top Matching Questions", lines=4, interactive=False)
        
        with gr.Tab("Answer Evaluation"):
            answer_input = gr.Textbox(label="Your Technical Answer", lines=5)
            eval_btn = gr.Button("Evaluate Answer", variant="primary")
            evaluation_output = gr.Markdown(label="Expert Evaluation")
        
        submit_btn.click(
            fn=process_resume,
            inputs=resume_input,
            outputs=[question_output, reference_output, matches_output]
        )
        
        eval_btn.click(
            fn=lambda q,a,r: evaluate_answer(q,a,r,tokenizer,phi2),
            inputs=[question_output, answer_input, reference_output],
            outputs=evaluation_output
        )
    
    return app

if __name__ == "__main__":
    try:
        print("Loading data...")
        data = load_data()
        
        print("Loading Phi-2...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        phi2 = AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-2",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
        
        print("Launching interface...")
        interface = create_interface(data, phi2, tokenizer)
        interface.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False
        )
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        input("Press Enter to exit...")