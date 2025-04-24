from flask import Flask, request, render_template, jsonify, redirect, url_for
import os
import json
from werkzeug.utils import secure_filename
from rag_engine import InterviewQuestionGenerator
from utils.similarity_checker import compute_detailed_evaluation
from utils.session_manager import create_session, load_session, save_session, append_question
from utils.metrics import evaluate_answer

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

    # if resume_file.filename == '':
    #     return "❌ No resume file selected"

    # filename = secure_filename(resume_file.filename)
    # resume_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    # resume_file.save(resume_path)

    # output_path = os.path.join(OUTPUT_FOLDER, 'qa_results.json')
    # result = generator.generate_and_save_qa(job_desc, resume_path, output_path)

    # return redirect(url_for('show_results'))
    if resume_file.filename == '':
        return "❌ No resume file selected"

    filename = secure_filename(resume_file.filename)
    resume_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    resume_file.save(resume_path)

    # ✅ Create session and redirect to interview
    session_id = create_session(job_desc, resume_path)
    return redirect(url_for('interview', session_id=session_id))

@app.route('/interview') 
def interview(): 
    session_id = request.args.get('session_id') 
    session = load_session(session_id)
    if not session:
        return redirect(url_for('index'))

    # First question
    if session["current_index"] == 0 and not session["questions_asked"]:
        output_path = os.path.join(OUTPUT_FOLDER, f"{session_id}_qa.json")
        result = generator.generate_and_save_qa(
            session["job_description"],
            session["resume_path"],
            output_path
        )
        session["questions_asked"].append({
            "q_no": 1,
            "question": result["generated_question"],
            "user_answer": "",
            "evaluation": None,
            "follow_up": False
        })
        session["current_index"] = 0
        save_session(session)

    question = session["questions_asked"][session["current_index"]]["question"]
    return render_template("interview.html", session_id=session_id, question=question, q_no=session["current_index"] + 1)


@app.route('/submit_answer', methods=['POST']) 
def submit_answer(): 
    session_id = request.form["session_id"] 
    user_answer = request.form["user_answer"]
    session = load_session(session_id)
    if not session:
        return redirect(url_for("index"))

    current_q = session["questions_asked"][session["current_index"]]
    model_answer = generator.retrieve_questions(current_q["question"])[0]["Cleaned_Answer"]
    evaluation = compute_detailed_evaluation(user_answer, model_answer)
    extra_scores = evaluate_answer(user_answer, model_answer)

    # Update answer and evaluation
    session["questions_asked"][session["current_index"]]["user_answer"] = user_answer
    session["questions_asked"][session["current_index"]]["evaluation"] = evaluation
    session["questions_asked"][session["current_index"]]["metrics"] = extra_scores

    session["current_index"] += 1

    # ✅ Generate next question if not reached limit
    if session["current_index"] < 5:
        output_path = os.path.join(OUTPUT_FOLDER, f"{session_id}_qa.json")
        result = generator.generate_and_save_qa(
            session["job_description"],
            session["resume_path"],
            output_path
        )
        session["questions_asked"].append({
            "q_no": session["current_index"] + 1,
            "question": result["generated_question"],
            "user_answer": "",
            "evaluation": None,
            "follow_up": False
        })

    save_session(session)

    # Redirect to summary after 5 questions
    if session["current_index"] >= 5:
        return redirect(url_for("summary", session_id=session_id))
    else:
        return redirect(url_for("interview", session_id=session_id))

   
@app.route('/summary') 
def summary(): 
    session_id = request.args.get("session_id") 
    session = load_session(session_id) 
    return render_template("summary.html", session=session)


@app.route('/end_interview', methods=['POST'])
def end_interview():
    session_id = request.form['session_id']
    session = load_session(session_id)
    if not session:
        return redirect(url_for("index"))

    session["interview_ended"] = True
    save_session(session)

    return redirect(url_for("report", session_id=session_id))

@app.route('/report')
def report():
    session_id = request.args.get("session_id")
    session = load_session(session_id)
    if not session:
        return redirect(url_for("index"))
    return render_template("report.html", session=session)


# @app.route('/results')
# def show_results():
#     output_path = os.path.join(OUTPUT_FOLDER, 'qa_results.json')
#     if os.path.exists(output_path):
#         with open(output_path, 'r') as f:
#             results = json.load(f)
#         return render_template('results.html', results=results)
#     return redirect(url_for('index'))

# @app.route('/compare', methods=['POST'])
# # def compare_answer():
# #     user_answer = request.form['user_answer']
# #     model_answer = request.form['model_answer']
# #     score = compute_similarity(user_answer, model_answer)
# #     return jsonify({'similarity_score': score})
# def compare_answer(): 
#     user_answer = request.form['user_answer'] 
#     model_answer = request.form['model_answer']
#     evaluation = compute_detailed_evaluation(user_answer, model_answer) 
#     return jsonify(evaluation)


if __name__ == '__main__':
    app.run(debug=True)