
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_similarity(answer1, answer2):
    emb1 = model.encode(answer1, convert_to_tensor=True)
    emb2 = model.encode(answer2, convert_to_tensor=True)
    score = util.pytorch_cos_sim(emb1, emb2).item()
    return score
