import re
import pysbd
import pandas as pd
from datasets import load_dataset

segmenter = pysbd.Segmenter(language="en", clean=False)

def load_data(dataset_name="ubaada/booksum-complete-cleaned", split="test"):
    """Loads dataset from HuggingFace."""
    ds = load_dataset(dataset_name)
    df = pd.DataFrame([{
        'book_id': f"Book {i.get('book_title', '')} {i.get('chapter_id', '')}",
        'text': i['text'], 
        'summary': i['summary'][0]['text'],
        'text_length': len(i['text']),
        'summary_length': len(i['summary'][0]['text'].split(' ')),
        'source' : i['summary'][0]['source'],
        'is_aggregate': i['is_aggregate'],
    } for i in ds[split]])

    return df

def preprocess_text(text: str) -> str:
    text = text.replace("\\'", "'")
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'\n{2,}', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = text.strip()
    text = text.replace("\\'", "'")
    text = re.sub(r'\(\*\)', '', text)
    text = re.sub(r'\(\+\)', '', text)

    text = re.sub(r'\(\*\)', '', text)
    text = re.sub(r'\(\+\)', '', text)

    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text) 
    text = re.sub(r'\n{2,}', ' ', text)         
    text = re.sub(r'\s{2,}', ' ', text)         
    text = text.strip()

    return text

def split_to_sentences(text: str):
    text = re.sub(r"\s+", " ", text).strip()
    sents = segmenter.segment(text)
    sents = [s for s in sents if any(ch.isalnum() for ch in s)]

    numbered_sents = [f"{i+1}. {s.strip()}" for i, s in enumerate(sents)]
    plain_sents    = [s.strip() for s in sents]
    return pd.Series([plain_sents, numbered_sents])
