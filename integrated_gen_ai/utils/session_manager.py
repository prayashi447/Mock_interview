from pathlib import Path
import json
import time
import uuid
import os

SESSIONS_DIR = Path("data/sessions")
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

def create_session(job_desc, resume_path):
    session_id = f"session_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    session_file = SESSIONS_DIR / f"{session_id}.json"
    session_data = {
        "session_id": session_id,
        "job_description": job_desc,
        "resume_path": str(resume_path),
        "questions_asked": [],
        "current_index": 0
    }
    with open(session_file, "w") as f:
        json.dump(session_data, f, indent=2)
    return session_id

def load_session(session_id):
    session_file = SESSIONS_DIR / f"{session_id}.json"
    if session_file.exists():
        with open(session_file, "r") as f:
            return json.load(f)
    return None

def save_session(session_data):
    session_file = SESSIONS_DIR / f"{session_data['session_id']}.json"
    with open(session_file, "w") as f:
        json.dump(session_data, f, indent=2)

def append_question(session_id, question, user_answer, evaluation, follow_up=False):
    session_data = load_session(session_id)
    if session_data:
        session_data["questions_asked"].append({
            "q_no": len(session_data["questions_asked"]) + 1,
            "question": question,
            "user_answer": user_answer,
            "evaluation": evaluation,
            "follow_up": follow_up
        })
        session_data["current_index"] += 1
        save_session(session_data)