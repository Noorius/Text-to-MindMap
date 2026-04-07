import math
import json
import ollama
from core_identifier import safe_json_decode_python

def rank_nodes(book_id, leaf_texts, conflict, chunk_size=30, target_summary_tokens=None):
    n = len(leaf_texts)
    
    if n == 0:
        return []
    
    if n <= chunk_size:
        selected_indices = select_events(book_id, leaf_texts, conflict, target_summary_tokens=target_summary_tokens, max_tokens=512)

        return [leaf_texts[i] for i in selected_indices]
    

    candidates = []
    n_chunks = math.ceil(n / chunk_size)
    
    for i in range(n_chunks):
        start = i * chunk_size
        end   = min(start + chunk_size, n)
        chunk = leaf_texts[start:end]
        
        local_indices = select_events(book_id, chunk, conflict, target_summary_tokens=target_summary_tokens, max_tokens=512)
        
        for local_idx in local_indices:
            global_idx = start + local_idx
            if global_idx not in [c['global_idx'] for c in candidates]:
                candidates.append({
                    'global_idx': global_idx,
                    'text': leaf_texts[global_idx]
                })
    
    if not candidates:
        step = max(1, n // 10)
        print("not", len(leaf_texts[::step]))
        return leaf_texts[::step]

    
    candidate_texts = [c['text'] for c in candidates]

    print('before', len(candidate_texts))
    
    if len(candidate_texts) <= chunk_size:
        final_local_indices = select_events(book_id, candidate_texts, conflict, target_summary_tokens=target_summary_tokens, max_tokens=512)
        print('less', len(final_local_indices))
        selected_texts = [candidate_texts[i] for i in final_local_indices]
    else:
        selected_texts = candidate_texts

    selected_words = sum(len(t.split()) for t in selected_texts)
    
    target_words = int(target_summary_tokens) if target_summary_tokens else 0
    
    total_original_words = sum(len(t.split()) for t in leaf_texts)
    
    if target_words > 0 and selected_words < (target_words * 1.5):

        if total_original_words > (target_words * 1.5):
            print(f"STARVATION PREVENTED! Target: {target_words} words. Needed raw: ~{int(target_words * 1.5)}. Got: {selected_words}. Fallback to uniform sampling.")
            
            words_needed = target_words * 2.0
            
            fallback_texts = []
            current_words = 0
            
            step = max(1, len(leaf_texts) // int(len(leaf_texts) * (words_needed / total_original_words)))
            
            for i in range(0, len(leaf_texts), step):
                fallback_texts.append(leaf_texts[i])
                current_words += len(leaf_texts[i].split())
                if current_words >= words_needed:
                    break
                    
            return fallback_texts

    return selected_texts

def select_events(book_name, sentences, conflict_json, target_summary_tokens=None, max_tokens=1000, model="mistral-small3.2:24b"): #mistral-large-3:675b-cloud
    """
    Agent 2: Selects the indices of sentences crucial to the identified conflict.
    """

    total_sentences = len(sentences)
    if target_summary_tokens is not None:
        target_words = int(target_summary_tokens)
        if target_words > 500:
            selection_rule = (
                f"We are writing a LONG, detailed summary ({target_words} words). "
                "Select roughly 50-70% of the passages: focus on scenes that introduce, escalate, or resolve the main conflict, "
                "and skip repetitive banter or minor asides."
            )
        elif target_words > 200:
            selection_rule = (
                f"We are writing a MEDIUM summary ({target_words} words). "
                "Select roughly 30-50% of the passages. Focus on core actions, major decisions, and key revelations."
            )
        else:
            selection_rule = f"We are writing a VERY SHORT summary ({target_words} words). Select ONLY the 3-5 most absolutely critical passages. Be ruthless. Drop all minor details."
    else:
        selection_rule = "Select the passages necessary to understand how this conflict appears, escalates, and resolves."
    
    selection_schema = {
        "type": "object",
        "properties": {
            "selected_indices": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Array of integer indices of the selected passages."
            }
        },
        "required": ["selected_indices"]
    }

    prompt = (
        f"You are selecting key events for a summary of a single chapter from '{book_name}'.\n"
        "You are provided with the identified core conflict and a numbered list of passages.\n\n"
        "SELECTION GOAL:\n"
        f"{selection_rule}\n\n"
        "INCLUDE:\n"
        "INCLUDE:\\n"
        "- The first event that introduces the conflict.\\n"
        "- All major decisions, confrontations, and reversals related to this conflict.\\n"
        "- If a passage introduces a new instrument for the conflict (a letter, threat, promise, deadline, weapon), include it.\\n"
        "- Always include the last passage if it contains the closing action, a departure, or the final image of the scene.\\n"
        "- When several passages describe one extended interaction, prefer 2–3 that cover both the physical turning points AND any explicit decisions or vows (promises, refusals, deadlines), and skip minor rephrasings of the same moment.\\n\\n"        
        
        "Identified Conflict:\n"
        f"{json.dumps(conflict_json, indent=2)}\n\n"
        f'''You must respond ONLY with a valid JSON object. 
        Do NOT use raw newlines inside JSON string values. Use \\n instead.
        Your JSON must strictly follow this exact schema:
        {json.dumps(selection_schema, indent=2)}'''
        "Passages:\n"
    )

    for ind, s in enumerate(sentences):
        prompt += f"[{ind}] {s}\n"    

    try:
        response = ollama.generate(
            model=model,
            prompt=prompt,
            think=False,
            keep_alive='2m',
            format='json',
            options={
                'temperature': 0,
                'num_predict': max_tokens,
                "num_ctx": 262144
            }
        )
        
        text = response.response.strip()
        parsed_json = safe_json_decode_python(text)
    except Exception as e:
        print(f"Selection generation error: {e}")
        parsed_json = None
    
    if parsed_json and "selected_indices" in parsed_json:
        valid_indices = sorted(list(set(
            [i for i in parsed_json["selected_indices"] if isinstance(i, int) and 0 <= i < len(sentences)]
        )))
    else:
        valid_indices = []
    
    if len(sentences) > 0 and (len(sentences) - 1) not in valid_indices:
        valid_indices.append(len(sentences) - 1)
        valid_indices.sort()

    if not valid_indices and len(sentences) > 0:
        step = max(1, len(sentences) // max(1, int(len(sentences) * 0.3)))
        valid_indices = list(range(0, len(sentences), step))

    return valid_indices