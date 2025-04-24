from evaluate import load
import bert_score

bleu = load("bleu")
rouge = load("rouge")

def evaluate_answer(pred, ref):
    results = {}

    try:
        bleu_score = bleu.compute(predictions=[pred], references=[ref])["bleu"]
        results["BLEU"] = round(bleu_score, 4)
    except:
        results["BLEU"] = 0.0

    try:
        rouge_score = rouge.compute(predictions=[pred], references=[ref])["rougeL"]
        results["ROUGE"] = round(rouge_score, 4)
    except:
        results["ROUGE"] = 0.0

    try:
        P, R, F1 = bert_score.score([pred], [ref], lang="en", verbose=False)
        results["BERTScore"] = round(F1[0].item(), 4)
    except:
        results["BERTScore"] = 0.0

    return results
