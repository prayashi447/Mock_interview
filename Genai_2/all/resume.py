import gradio as gr
import re
import docx
from PyPDF2 import PdfReader

# Comprehensive skills database
SKILLS_DATABASE = {
    # Technical Skills
    'Programming Languages': [
        'python', 'java', 'c++', 'javascript', 'typescript', 'ruby', 'swift', 
        'kotlin', 'scala', 'go', 'rust', 'php', 'perl', 'r', 'matlab'
    ],
    'Web Technologies': [
        'html', 'css', 'react', 'angular', 'vue', 'nodejs', 'express', 
        'django', 'flask', 'spring', 'asp.net', 'ruby on rails'
    ],
    'Databases': [
        'mysql', 'mongodb', 'postgresql', 'sqlite', 'oracle', 'redis', 
        'dynamodb', 'cassandra', 'firebase'
    ],
    'Cloud Platforms': [
        'aws', 'azure', 'google cloud', 'heroku', 'digital ocean', 
        'aws lambda', 'google cloud platform', 'azure devops'
    ],
    'DevOps & Tools': [
        'docker', 'kubernetes', 'jenkins', 'git', 'github', 'gitlab', 
        'ansible', 'terraform', 'CI/CD', 'maven', 'gradle'
    ],
    'Machine Learning & AI': [
        'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'numpy', 
        'pandas', 'machine learning', 'deep learning', 'natural language processing', 
        'computer vision', 'data science'
    ],
    'Mobile Development': [
        'android', 'ios', 'react native', 'flutter', 'xamarin', 'swift', 'kotlin'
    ],
    'Soft Skills': [
        'communication', 'leadership', 'teamwork', 'problem solving', 
        'critical thinking', 'time management', 'adaptability'
    ]
}

def extract_skills(file):
    # Read the file content
    text = ""
    if file.name.endswith('.pdf'):
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    elif file.name.endswith('.docx'):
        doc = docx.Document(file)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    else:
        text = file.decode('utf-8')
    
    # Normalize text
    text = text.lower()
    
    # Extract skills
    found_skills = {}
    
    # Iterate through skill categories
    for category, skills in SKILLS_DATABASE.items():
        category_skills = []
        for skill in skills:
            # Use word boundary regex to match whole words
            pattern = r'\b' + re.escape(skill.lower()) + r'\b'
            if re.search(pattern, text):
                category_skills.append(skill)
        
        if category_skills:
            found_skills[category] = category_skills
    
    # Format the output
    output = []
    for category, skills in found_skills.items():
        output.append(f"{category}:")
        for skill in skills:
            output.append(f"  - {skill}")
    
    return "\n".join(output) if output else "No skills found"

# Create Gradio interface
def create_skills_extractor():
    iface = gr.Interface(
        fn=extract_skills,
        inputs=gr.File(label="Upload Resume (PDF, DOCX, TXT)"),
        outputs=gr.Textbox(label="Extracted Skills"),
        title="Resume Skills Extractor",
        description="Upload your resume to extract professional skills"
    )
    return iface

# Launch the interface
if __name__ == "__main__":
    demo = create_skills_extractor()
    demo.launch()