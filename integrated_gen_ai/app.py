from flask import Flask, request, render_template, jsonify, redirect, url_for
import os
import json
from werkzeug.utils import secure_filename
from rag_engine import InterviewQuestionGenerator
from utils.similarity_checker import compute_similarity

app = Flask(__name__)

UPLOAD_FOLDER = 'data/raw'
OUTPUT_FOLDER = 'data/output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

generator = InterviewQuestionGenerator()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    job_desc = request.form.get('job_desc', '')
    resume_file = request.files['resume']

    if resume_file.filename == '':
        return "❌ No resume file selected"

    filename = secure_filename(resume_file.filename)
    resume_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    resume_file.save(resume_path)

    output_path = os.path.join(OUTPUT_FOLDER, 'qa_results.json')
    result = generator.generate_and_save_qa(job_desc, resume_path, output_path)

    return redirect(url_for('show_results'))

@app.route('/results')
def show_results():
    output_path = os.path.join(OUTPUT_FOLDER, 'qa_results.json')
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            results = json.load(f)
        return render_template('results.html', results=results)
    return redirect(url_for('index'))

@app.route('/compare', methods=['POST'])
def compare_answer():
    user_answer = request.form['user_answer']
    model_answer = request.form['model_answer']
    score = compute_similarity(user_answer, model_answer)
    return jsonify({'similarity_score': score})

if __name__ == '__main__':
    app.run(debug=True)