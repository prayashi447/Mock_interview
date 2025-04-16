import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import re
import docx
from PyPDF2 import PdfReader

class InterviewQuestionGenerator:
    def __init__(self):
        model_name = "microsoft/phi-2"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
            self.generator = pipeline('text-generation', model=self.model, tokenizer=self.tokenizer, max_length=150, num_return_sequences=1, do_sample=True, temperature=0.7)
        except Exception as e:
            print(f"Model loading error: {e}")
            self.generator = None

    def read_resume(self, file):
        """Read resume content from PDF, DOCX, or TXT."""
        text = ""
        if file.name.endswith('.pdf'):
            pdf_reader = PdfReader(file)
            text = "\n".join(page.extract_text() or "" for page in pdf_reader.pages)
        elif file.name.endswith('.docx'):
            doc = docx.Document(file)
            text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
        else:
            text = file.read().decode('utf-8')
        return text.strip()

    def extract_skills(self, text):
        """Extract skills from resume text using regex for better matching."""
        SKILLS_DATABASE = {
            'Programming Languages': ['python', 'java', 'c\+\+', 'javascript', 'typescript', 'ruby', 'swift', 'kotlin', 'scala', 'go', 'rust', 'php', 'perl', 'r', 'matlab'],
            'Web Technologies': ['html', 'css', 'react', 'angular', 'vue', 'nodejs', 'express', 'django', 'flask', 'spring', 'asp\.net', 'ruby on rails'],
            'Databases': ['mysql', 'mongodb', 'postgresql', 'sqlite', 'oracle', 'redis', 'dynamodb', 'cassandra', 'firebase'],
            'Cloud Platforms': ['aws', 'azure', 'google cloud', 'heroku', 'digital ocean', 'aws lambda', 'gcp', 'azure devops'],
            'DevOps & Tools': ['docker', 'kubernetes', 'jenkins', 'git', 'github', 'gitlab', 'ansible', 'terraform', 'ci/cd', 'maven', 'gradle'],
            'Machine Learning & AI': ['tensorflow', 'pytorch', 'keras', 'scikit-learn', 'numpy', 'pandas', 'machine learning', 'deep learning', 'nlp', 'computer vision', 'data science'],
            'Mobile Development': ['android', 'ios', 'react native', 'flutter', 'xamarin', 'swift', 'kotlin'],
            'Soft Skills': ['communication', 'leadership', 'teamwork', 'problem solving', 'critical thinking', 'time management', 'adaptability']
        }

        found_skills = set()
        text_lower = text.lower()
        for category, skills in SKILLS_DATABASE.items():
            for skill in skills:
                if re.search(rf'\b{skill}\b', text_lower):
                    found_skills.add(skill.replace('\\', ''))  # Remove escape sequences for better display
        return list(found_skills)

    def generate_questions(self, skills):
        """Generate interview questions using AI model."""
        if not self.generator:
            return "Error: AI model could not be loaded."
        if not skills:
            return "No skills found to generate questions."

        generated_questions = []
        for skill in skills:
            prompt = f"Generate a detailed technical interview question about {skill}. The question should assess both theoretical understanding and practical application."
            try:
                response = self.generator(prompt, max_length=150, num_return_sequences=1)
                question = response[0]['generated_text'].replace(prompt, '').strip()
                question = re.sub(r'^[\n\s]+|[\n\s]+$', '', question).capitalize()
                if question:
                    generated_questions.append(f"Question about {skill}: {question}")
            except Exception as e:
                print(f"Error generating question for {skill}: {e}")
        return "\n\n".join(generated_questions)

    def generate_interview_questions(self, file):
        """Main method to generate interview questions."""
        resume_text = self.read_resume(file)
        skills = self.extract_skills(resume_text)
        return self.generate_questions(skills)

# Create Gradio interface
def create_interview_question_generator():
    question_generator = InterviewQuestionGenerator()
    iface = gr.Interface(
        fn=question_generator.generate_interview_questions,
        inputs=gr.File(label="Upload Resume (PDF, DOCX, TXT)"),
        outputs=gr.Textbox(label="AI-Generated Interview Questions"),
        title="AI Interview Question Generator",
        description="Upload your resume to generate AI-powered interview questions based on your skills."
    )
    return iface

if __name__ == "__main__":
    demo = create_interview_question_generator()
    demo.launch()
