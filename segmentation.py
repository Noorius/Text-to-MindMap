import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

def get_embedder(model_name="sentence-transformers/all-mpnet-base-v2"):
    return SentenceTransformer(model_name, trust_remote_code=True)

def embed_sentences(embedder, sentences, batch_size=32):
    return embedder.encode(sentences, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)

def cosine_similarity(embeddings):
    tensor_embeddings = torch.tensor(embeddings)
    return F.cosine_similarity(tensor_embeddings.unsqueeze(1), tensor_embeddings.unsqueeze(0), dim=-1).numpy()

def adaptive_segmentation(sentences, embeddings, target_segments, k=0.9):
    n = len(sentences)
    boundaries = [0]  
    sims = (cosine_similarity(embeddings) + 1) / 2 
    window = int(max(3, min(10, len(sentences)*0.05)))

    segments = []
    gap_score = []
    for i in range(1, n):
        lo = max(0, i - window)
        hi = min(n-1, i + window)
        local_pairs = []
        for j in range(lo+1, hi+1):
            local_pairs.append(sims[j-1, j])
        mu = np.mean(local_pairs)
        sigma = np.std(local_pairs) if np.std(local_pairs)>1e-6 else 1e-6
        cur = sims[i-1, i]
        if cur < mu - k * sigma:
            if i - boundaries[-1] >= 5:
                boundaries.append(i)
                gap_score.append((i ,(mu - cur) / sigma))
    
    for a, b in zip(boundaries, boundaries[1:]+[n]):
       segments.append(list(range(a, b)))

    if target_segments is None or len(segments) <= target_segments:
        return segments

    boundary_scores = {pos: score for pos, score in gap_score}
    segments_indices = [(seg[0], seg[-1] + 1) for seg in segments]

    while len(segments_indices) > target_segments and len(segments_indices) > 1:
        candidates = []
        for i in range(len(segments_indices) - 1):
            boundary_pos = segments_indices[i][1]  
            score = boundary_scores.get(boundary_pos, 0.0)  
            seg_len = segments_indices[i][1] - segments_indices[i][0] \
                      + segments_indices[i+1][1] - segments_indices[i+1][0]
            candidates.append((score, seg_len, i))

        candidates.sort(key=lambda x: (x[0], x[1]))
        _, _, best_i = candidates[0]

        left = segments_indices[best_i]
        right = segments_indices[best_i + 1]
        merged = (left[0], right[1])

        segments_indices = (
            segments_indices[:best_i] + [merged] + segments_indices[best_i + 2:]
        )

    final_segments = [list(range(start, end)) for start, end in segments_indices]
    return final_segments

def adaptive_segmentation_2(sentences, embeddings, k=0.9):
    n = len(sentences)
    boundaries = [0]  
    sims = (cosine_similarity(embeddings) + 1) / 2 
    window = int(max(3, min(10, len(sentences)*0.05)))

    segments = []
    gap_score = []
    for i in range(1, n):
        lo = max(0, i - window)
        hi = min(n-1, i + window)
        local_pairs = []
        for j in range(lo+1, hi+1):
            local_pairs.append(sims[j-1, j])
        mu = np.mean(local_pairs)
        sigma = np.std(local_pairs) if np.std(local_pairs)>1e-6 else 1e-6
        cur = sims[i-1, i]
        if cur < mu - k * sigma:
            if i - boundaries[-1] >= 5:
                boundaries.append(i)
                gap_score.append((i ,(mu - cur) / sigma))
    
    for a, b in zip(boundaries, boundaries[1:]+[n]):
       segments.append(list(range(a, b)))
    return segments
