from pathlib import Path
import numpy as np
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer, util
from sklearn.neighbors import NearestNeighbors
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from termcolor import colored
from collections import defaultdict
import csv 
import spacy
import fitz  # PyMuPDF

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DATA_PATH = DATA_DIR / "raw" / "questions.csv"
EMBEDDING_MODEL = 'BAAI/bge-large-en-v1.5'
LLM_MODEL = 'microsoft/phi-2'

encoder = SentenceTransformer(EMBEDDING_MODEL)

class InterviewQuestionGenerator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.load_data()
        self.load_models()
        self.build_index()
        self.nlp = spacy.load("en_core_web_sm")

    def load_data(self):
        print(colored("Loading embeddings and metadata...", "cyan"))
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
        print(colored(f"Loading language model {LLM_MODEL}...", "cyan"))
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, trust_remote_code=True, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.llm = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            trust_remote_code=True
        ).to(self.device)

    def build_index(self):
        print(colored("Building search indexes...", "cyan"))
        self.question_index = NearestNeighbors(n_neighbors=5, metric='cosine')
        self.question_index.fit(self.question_embeddings)
        self.category_index = NearestNeighbors(n_neighbors=1, metric='cosine')
        self.category_index.fit(self.category_embeddings_meta)

    def process_resume(self, pdf_path, threshold=0.7):
        print(colored(f"[INFO] Processing resume from: {pdf_path}", "cyan"))
        RESUME_SKILL_CATEGORIES = {
            "DevOps": ["GitHub", "Git", "Shell Scripting"],
            "Containers and Cloud": ["Kubernetes", "Firebase", "AWS", "Docker"],
            "SQL": ["MySQL", "SQL", "SQL Server", "MS Access", "Oracle", "DB2"],
            "General Software Engineering": ["MERN", "Flutter", "JavaScript", "CSS"],
            "AI (Data Science)": ["AI", "ML", "Deep Learning", "NLP", "Machine Learning"]
        }
        doc = fitz.open(pdf_path)
        text = " ".join([page.get_text("text") for page in doc]).strip()
        doc = self.nlp(text.lower())
        tokens = list(set([token.text for token in doc if token.is_alpha]))

        categorized = defaultdict(set)
        skill_model = SentenceTransformer("all-MiniLM-L6-v2")

        category_embeddings = {
            cat: skill_model.encode(skills, convert_to_tensor=True)
            for cat, skills in RESUME_SKILL_CATEGORIES.items()
        }

        for token in tokens:
            token_embedding = skill_model.encode(token, convert_to_tensor=True)
            best_category = None
            best_score = 0
            for category, embeddings in category_embeddings.items():
                sim_score = util.cos_sim(token_embedding, embeddings).max().item()
                if sim_score >= threshold and sim_score > best_score:
                    best_category = category
                    best_score = sim_score
            if best_category:
                categorized[best_category].add(token)
        return {cat: sorted(list(skills)) for cat, skills in categorized.items()}

    def get_category(self, query):
        query_embedding = encoder.encode(query, normalize_embeddings=True)
        _, indices = self.category_index.kneighbors([query_embedding])
        category_idx = indices[0][0]
        return self.category_mapping[category_idx]

    def retrieve_questions(self, query, category=None, n=1):
        query_embedding = encoder.encode(query, normalize_embeddings=True)
        if category:
            cat_questions = [i for i, m in enumerate(self.metadata) if m['Category'] == category]
            filtered_embeddings = self.question_embeddings[cat_questions]
            temp_index = NearestNeighbors(n_neighbors=min(n, len(cat_questions)), metric='cosine')
            temp_index.fit(filtered_embeddings)
            _, indices = temp_index.kneighbors([query_embedding])
            original_indices = [cat_questions[i] for i in indices[0]]
        else:
            _, indices = self.question_index.kneighbors([query_embedding])
            original_indices = indices[0][:n]
        return [self.df.iloc[i] for i in original_indices]

    def generate_question_and_answer(self, context, category, temperature=0.7, max_new_tokens=200):
        prompt = f"""You are a technical interviewer. Based on the context below, generate a new technical interview question about {category}, followed by its ideal answer.

Context:
{context}

New Question and Answer:
Q:"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            return_attention_mask=True,
            truncation=True,
            max_length=2048
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
        ).strip()

        return generated_text

    def generate_and_save_qa(self, job_desc, resume_path, output_path):
        context_text = job_desc.strip()
        skills = self.process_resume(resume_path)
        for cat, skill_list in skills.items():
            context_text += f"\n[{cat}] " + ", ".join(skill_list)

        category = self.get_category(context_text)
        retrieved = self.retrieve_questions(context_text, category, n=1)
        q = retrieved[0]

        combined = f"{context_text}\nExample Q: {q['Cleaned_Question']}\nExample A: {q['Cleaned_Answer']}"
        qa_pair = self.generate_question_and_answer(combined, category)

        # Split model output into question and answer
        if "A:" in qa_pair:
            generated_question, generated_answer = qa_pair.split("A:", 1)
        else:
            generated_question = qa_pair
            generated_answer = ""

        # Save only the question to output for frontend
        result = {
            "category": category,
            "retrieved_question": q.to_dict(),
            "generated_question": generated_question.strip()
        }

        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)

        # Save the full Q&A to a CSV for backend
        log_path = Path(output_path).parent / "qa_answer_log.csv"
        write_header = not log_path.exists()

        with open(log_path, "a", newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            if write_header:
                writer.writerow(["Category", "Question", "Answer"])
            writer.writerow([category, generated_question.strip(), generated_answer.strip()])

        # Print the answer to terminal
        print("\n🧠 Model Answer (hidden from frontend):")
        print(generated_answer.strip())

        return result

    def rag_pipeline(self, topic, category=None):
        if not category:
            category = self.get_category(topic)
            print(colored(f"Automatically determined category: {category}", "yellow"))

        retrieved = self.retrieve_questions(topic, category, n=1)
        q = retrieved[0]

        print(colored("\nRetrieved Question:", "blue"))
        print(colored(f"- {q['Cleaned_Question']}", "green"))
        print(colored(f"  Answer: {q['Cleaned_Answer'][:200]}...\n", "magenta"))

        context = f"Question: {q['Cleaned_Question']}\nAnswer: {q['Cleaned_Answer']}"
        qa_pair = self.generate_question_and_answer(context, category)

        print(colored("Generated Question and Answer:", "blue"))
        print(colored(qa_pair, "green"))

        return {
            "category": category,
            "retrieved_question": q,
            "generated_qa": qa_pair
        }