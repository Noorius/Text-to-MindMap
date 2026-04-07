import json
import ollama
from core_identifier import safe_json_decode_python
import time

def polish(book_name, selected_sentences, target_words=100, model="mistral-small3.2:24b", is_reduce_step=False, style='a neutral, academic'):
    """
    Agent 3: Writes the final summary using ONLY the selected sentences.
    """
    target_words = int(target_words)
    
    if not is_reduce_step:
        action_verb = "SUMMARIZE"
        if target_words < 150:
            fact_rule = f"- Combine the provided passages into a concise summary of maximum {target_words} words.\\\\n"
        else:
            fact_rule = (
                f"- You do NOT have to preserve every minor detail from every passage. "
                f"Prioritize core plot actions, key decisions, and turning points. "
                f"Drop repetitive or purely atmospheric details if needed.\\\\n"
            )
    else:
        current_input_words = sum(len(s.split()) for s in selected_sentences)
        
        if current_input_words <= (target_words * 1.2):
            action_verb = "SMOOTH AND MERGE"
            fact_rule = (
                "- MERGE the provided texts into a single cohesive narrative.\\n"
                "- DO NOT add any new details. DO NOT compress the text.\\n"
                f"- The output must remain exactly around {current_input_words} words.\\n"
            )
        else:
            action_verb = "SUMMARIZE"
            fact_rule = (
                f"- Synthesize the texts into a narrative of strictly between {int(target_words * 0.8)} and {target_words} words.\\n"
                "- You MUST drop minor details to hit this strict word limit, but keep the core plot intact.\\n"
            )

    summary_schema = {
        "type": "object",
        "properties": {"summary": {"type": "string"}}
    }
    
    
    prompt = (
        f"Here are the key events from a chapter of '{book_name}'.\\n"
        f"{action_verb} them into a text of strictly {target_words} words. "
        f"Write in {style} summary style.\\n\\n"
        "RULES:\\n"
        f"{fact_rule}"
        "- Use ONLY indirect speech. Do NOT use quotation marks and do NOT reproduce dialogue verbatim.\\n"
        "- Do NOT use slang, colloquial phrases, or dramatic idioms "
        "(e.g., 'drops a bombshell', 'things get real', 'super religious').\\n"
        "- If a passage describes a physical conflict, confrontation, or use of a weapon — include it explicitly.\\n"
        "- If a passage contains a day/time word (Sunday, Wednesday, tomorrow) or a proper name/title, keep it, "
        "but paraphrase the sentence in indirect speech.\\n"
        "- STRICTLY FORBIDDEN: Do NOT add any literary analysis or thematic observations. Only describe what happens to the characters.\\n"
        "- Connect the events smoothly using simple, formal transitions ('afterward', 'then', 'later') without over-detailing every step.\\n"
        "INSTRUCTIONS:\\n"
        "Return ONLY a valid JSON object containing the summary string. Do not include markdown code blocks or explanations.\\n\\n"
        f'''You must respond ONLY with a valid JSON object. 
        Do NOT use raw newlines inside JSON string values. Use \\n instead.
        Your JSON must strictly follow this exact schema:
        {json.dumps(summary_schema, indent=2)}'''
        "Passages:\\n"
    )

    for s in selected_sentences:
        prompt += "- " + s + "\\n"

    max_tokens_for_ollama = int(target_words * 1.5) + 150

    retries = 3
    for attempt in range(retries):
        try:
            if attempt > 0:
                max_tokens_for_ollama = max_tokens_for_ollama * 1.5
                
            response = ollama.generate(
                model=model, 
                prompt=prompt, 
                think=False, 
                keep_alive='2m',
                format='json',
                options={
                    'temperature': 0,
                    'num_predict': max_tokens_for_ollama,
                    "num_ctx": 262144
                }
            )
            output_text = response.response.strip()
            parsed_json = safe_json_decode_python(output_text)
            
            if parsed_json and "summary" in parsed_json:
                generated_words = len(parsed_json["summary"].split(' '))
                print(f"Generated words: {generated_words} / Target: {target_words}")
                return parsed_json["summary"]
                
        except Exception as e:
            if attempt < retries - 1:
                print(f"Ollama error: {e}. Retrying in 5 seconds... (Attempt {attempt + 1}/{retries})")
                time.sleep(5)  
            else:
                print("Max retries reached. Returning empty string.")
                return ""

    print('Error: JSON parsing failed')
    return output_text.replace('```json', '').replace('```', '').strip()


def polish1(index_num, book_name, selected_sentences, target_words=100, model="mistral-small3.2:24b", batch_size=7, is_reduce_step=False, style='CliffNotes or SparkNotes'):
    print('=================================================')
    print('Index', index_num)
    
    target_words = int(target_words)
    
    if len(selected_sentences) <= batch_size:
        print('Base', len(selected_sentences), batch_size)
        return polish(book_name, selected_sentences, target_words, model, is_reduce_step, style=style)
    
    total_input_words = sum(len(s.split()) for s in selected_sentences)
    
    compression_ratio = min(1.0, target_words / total_input_words) if total_input_words > 0 else 1.0
    
    intermediate_summaries = []
    
    for i in range(0, len(selected_sentences), batch_size):
        batch = selected_sentences[i:i + batch_size]
        batch_words = sum(len(s.split()) for s in batch)
        
        batch_target_words = int(batch_words * compression_ratio * 1.5)
        batch_target_words = max(50, batch_target_words)
        
        sub_summary = polish(
            book_name=book_name, 
            selected_sentences=batch, 
            target_words=batch_target_words, 
            model=model,
            is_reduce_step=False,
            style=style
        )
        intermediate_summaries.append(sub_summary)

    intermediate_summaries_len = len(" ".join(intermediate_summaries).split(' '))

    print('Not base: Generated', intermediate_summaries_len, '/ Global target', target_words)
    
    if 0.8 <= (intermediate_summaries_len / target_words) <= 1.2:
        print('condition 1 returned (within 20% margin)')
        return " ".join(intermediate_summaries)
        
    if intermediate_summaries_len <= target_words:
        print('condition 2 returned (text is already shorter than target)')
        return " ".join(intermediate_summaries)
        
    print('continued recursion')
    return polish1(
        index_num,
        book_name=book_name, 
        selected_sentences=intermediate_summaries, 
        target_words=target_words,
        model=model, 
        batch_size=batch_size,
        is_reduce_step=True,
        style=style
    )