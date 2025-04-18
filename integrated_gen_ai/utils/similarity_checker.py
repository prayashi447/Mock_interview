from sentence_transformers import SentenceTransformer, util
import re
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_detailed_evaluation(user_answer, model_answer):
    # Embedding similarity (Relevance)
    emb1 = model.encode(user_answer, convert_to_tensor=True)
    emb2 = model.encode(model_answer, convert_to_tensor=True)
    relevance_score = util.pytorch_cos_sim(emb1, emb2).item()

    # Coverage: keyword overlap
    user_tokens = set(re.findall(r'\w+', user_answer.lower()))
    model_tokens = set(re.findall(r'\w+', model_answer.lower()))
    common_tokens = user_tokens.intersection(model_tokens)
    coverage_score = len(common_tokens) / max(len(model_tokens), 1)

    # Fluency: based on filler word count and sentence quality
    filler_words = ["um", "uh", "like", "you know", "actually"]
    filler_count = sum(user_answer.lower().count(f) for f in filler_words)
    sentence_count = max(user_answer.count('.'), 1)
    avg_sentence_len = len(user_answer.split()) / sentence_count
    fluency_score = max(0.1, min(1.0, 1 - (filler_count * 0.05 + max(0, (5 - avg_sentence_len)) * 0.02)))

    # Overall score: average of individual scores
    overall_score = np.mean([relevance_score, coverage_score, fluency_score])

    return {
        "relevance": round(relevance_score, 2),
        "coverage": round(coverage_score, 2),
        "fluency": round(fluency_score, 2),
        "overall_score": round(overall_score, 2)
    }