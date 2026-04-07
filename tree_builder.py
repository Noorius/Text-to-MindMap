import re
import ollama
from src.segmentation import adaptive_segmentation_2
import time

def collect_all_nodes(tree, node_id_start=0):
    texts = []

    def _traverse(node):
        text = node.get("summary", None)
        leaf = node.get("leaf", False)
        if leaf and (text is not None):
            texts.append(f'{text}')

        for child in node.get("children", []):
            _traverse(child)

    _traverse(tree)
    return texts

def safe_json_decode_python(s: str) -> str:
    if not s:
        return ""
    m = re.search(r'summary\s*:\s*"(.+?)"', s, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    return s.strip()

def prepare_segments_for_tree_2(sentences, sentence_embeddings, segments):
    nodes_sentences = []
    nodes_embeddings = []
    for seg in segments:
        seg_sents = [sentences[i] for i in seg]
        seg_embs = [sentence_embeddings[i] for i in seg]
        nodes_sentences.append(seg_sents)
        nodes_embeddings.append(seg_embs)
    return nodes_sentences, nodes_embeddings, len(nodes_sentences)

def summarize(sentences, book_id, max_tokens=1000, model="mistral-small"):
    format_hint = 'summary: "..."'
    
    prompt = (
        f"The '{book_id}'.\n"
        "You are an expert literary editor.\n"
        "Task: rewrite the following sentences as ONE coherent narrative paragraph.\n"
        "RULES:\n"
        "- Describe only concrete events, actions, and character interactions from this text.\n"
        "- Convert all dialogue into indirect speech. Do NOT use quotation marks or verbatim quotes.\n"
        "- Do NOT comment on the text (no phrases like 'no decisions are made', 'no questions are asked').\n"
        "- Resolve pronouns: use explicit character names whenever possible instead of 'he', 'she', 'they'.\n"
        "- Do NOT add new events or interpretations; just condense what is there.\n\n"
        f"Answer in the format: {format_hint}\n\n"
        "Sentences:\n"
    )

    for s in sentences:
        prompt += "- " + s + "\n"
    prompt += "\nSummary:"

    retries = 3
    for attempt in range(retries):
        try:
            response = ollama.generate(
                model=model, 
                prompt=prompt, 
                think = False, 
                keep_alive='2m',
                options = {
                    'temperature': 0,
                    "num_ctx": 8192
                }
            )
            break
        except Exception as e:
            if attempt < retries - 1:
                print(f"Ollama error: {e}. Retrying in 5 seconds... (Attempt {attempt + 1}/{retries})")
                time.sleep(5)
            else:
                print("Max retries reached. Returning empty string.")
                return ""

    
    summary = response.response.strip()
    try:
        summary = safe_json_decode_python(summary)
    except Exception as e:
        print(e)
        return summary
    
    return summary

def recursive_tree_build(nodes_sentences, nodes_embeddings, book_id, max_leaf_size=7, depth=0, max_depth=2):

    n = len(nodes_sentences)
    if depth >= max_depth or sum(len(s) for s in nodes_sentences) <= max_leaf_size:
        summary = summarize([sent for s in nodes_sentences for sent in s], book_id)
        
        return {
                "depth": depth,
                "leaf": True,
                "summary": summary
                }
        
    children = []
    summaries = []
    for sents, embs in zip(nodes_sentences, nodes_embeddings):
        segments_ = adaptive_segmentation_2(sents, embs)
        segments_sentences_, segments_embeddings_, _ = prepare_segments_for_tree_2(sents, embs, segments_)

        if len(segments_) == 1 and len(segments_[0]) == len(sents):
            summary = summarize(sents, book_id)
            children.append({
                "depth": depth + 1,
                "leaf": True,
                "summary": summary
            })
            continue

        child = recursive_tree_build(
            nodes_sentences=segments_sentences_,
            nodes_embeddings=segments_embeddings_,
            book_id=book_id,
            max_leaf_size=max_leaf_size,
            depth=depth + 1,
            max_depth=max_depth
        )
        children.append(child)

    flat_texts = [child['summary'] for child in children]

    summary_text = ""
    
    node = {
        "depth": depth,
        "summary": summary_text,
        "children": children
    }
    
    return node
