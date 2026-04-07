import evaluate
import torch
import gc
from bert_score import BERTScorer

def scored_in_batches(scorer, preds, refs, batch_size=1):
    all_p, all_r, all_f1 = [], [], []
    
    with torch.no_grad():
        for i in range(0, len(preds), batch_size):
            p, r, f1 = scorer.score(preds[i:i+batch_size], refs[i:i+batch_size])
            all_p.append(p)
            all_r.append(r)
            all_f1.append(f1)
            
            torch.cuda.empty_cache()
            
    return torch.cat(all_p), torch.cat(all_r), torch.cat(all_f1)

def compute_rouge(predictions, references):
    rouge = evaluate.load('rouge')
    rouge_scores = rouge.compute(predictions=predictions, references=references)
    geometric_mean = (rouge_scores['rouge1'] * rouge_scores['rouge2'] * rouge_scores['rougeL']) ** (1/3)
    return rouge_scores['rouge1'], rouge_scores['rouge2'], rouge_scores['rougeL'], geometric_mean

def compute_bertscore(predictions, references, lang="en", batch_size=4):

    scorer = BERTScorer(
        model_type='microsoft/deberta-xlarge-mnli',
        num_layers=18,
        rescale_with_baseline=False,
        lang='en',
        device='cuda'
    )

    P, R, F1 = scored_in_batches(
            scorer, 
            predictions, 
            references, 
            batch_size=1
        )
    P, R, F1 = float(P.mean()), float(R.mean()), float(F1.mean())
    torch.cuda.empty_cache()
    gc.collect()

    return P, R, F1
