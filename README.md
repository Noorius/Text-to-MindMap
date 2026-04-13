# Hierarchical Multi-agent Extractive Framework For Long-document Mind Map Generation 

This the code for the article "Hierarchical Multi-agent Extractive Framework For Long-document Mind Map Generation"

<img src="/method_overview.png" width="500">

Datasets:
- GovReport: https://huggingface.co/datasets/ccdv/govreport-summarization
- MovieSum: https://huggingface.co/datasets/rohitsaxena/MovieSum
- BookSum: https://huggingface.co/datasets/ubaada/booksum-complete-cleaned
- QMSum: https://huggingface.co/datasets/MocktaiLEngineer/qmsum-processed
- MENSA: https://huggingface.co/datasets/rohitsaxena/MENSA
- SummScreenFD: https://huggingface.co/datasets/YuanPJ/summ_screen

Prompts:

| Agent Component | Prompt Template & Rules |
| :--- | :--- |
| **Core Identifier** <br> *(Local Chunk Extraction)* | **Task:** Analyze this specific excerpt from `{book_name}`. Identify the literal narrative events happening IN THIS EXCERPT ONLY. <br> **Rules:** <br> - DO NOT write a thematic essay or literary analysis. <br> - Describe only physical actions, character interactions, or specific plot points. <br> - If the text is merely a list or purely philosophical without character action, state: `No narrative events in this excerpt.` <br> **Output:** JSON object with `analysis_scratchpad`, `core_conflict` (1-2 sentences), and `sub_conflicts` (up to 3 items). |
| **Core Identifier** <br> *(Global Reduce Step)* | **Task:** You are given a list of literal events. Select the 4 most important physical events from this list that span the beginning, middle, and end of the story. <br> **Rules:** <br> - DO NOT write a summary or essay. DO NOT use abstract words. <br> - Just copy or slightly rephrase the concrete actions. <br> **Output:** JSON object with `top_4_events` array. |
| **Node Selector** <br> *(Filtering)* | **Task:** Select key events for a summary of a single chapter. You are provided with the identified core conflict and a numbered list of passages. <br> **Selection Goal:** {dynamic_rule_based_on_target_length}. <br> **Include:** <br> - The first event that introduces the conflict. <br> - All major decisions, confrontations, and reversals related to this conflict. <br> - Always include the last passage if it contains the closing action. <br> **Output:** JSON object with `selected_indices` (integer array). |
| **Length-Controlled Summarizer** <br> *(Generation)* | **Task:** Here are the key events from a chapter. {SUMMARIZE \| SMOOTH AND MERGE} them into a text of strictly {target_words} words. Write in a neutral, academic summary style. <br> **Rules:** <br> - Use ONLY indirect speech. Do NOT use quotation marks and do NOT reproduce dialogue verbatim. <br> - Do NOT use slang, colloquial phrases, or dramatic idioms. <br> - STRICTLY FORBIDDEN: Do NOT add any literary analysis or thematic observations. Only describe what happens to the characters. <br> - Connect the events smoothly using simple, formal transitions. <br> **Output:** JSON object with `summary` string. |

Zhetessov Nur
