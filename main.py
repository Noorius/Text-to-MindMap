import argparse
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
from src.data_loader import load_data, preprocess_text, split_to_sentences
from src.segmentation import get_embedder, embed_sentences, adaptive_segmentation_2, prepare_segments_for_tree_2
from src.tree_builder import recursive_tree_build, collect_all_nodes
from src.evaluation import compute_rouge, compute_bertscore
from core_identifier import identify_conflict
from event_selector import rank_nodes
from final_summary import polish1

def main(args):
    print(f"Loading dataset {args.dataset}...")
    df = load_data(args.dataset)

    embedder = get_embedder(args.embed_model)

    print("Preprocessing and chunking...")
    tqdm.pandas()
    df['clean_text'] = df['text'].progress_apply(preprocess_text)
    df[['sentences', 'numbered_sentences']] = df['clean_text'].progress_apply(
        lambda x: pd.Series(split_to_sentences(x))
    )

    print("Embedding sentences...")
    df['embeddings'] = df['sentences'].progress_apply(lambda x: embed_sentences(embedder, x))

    print("Adaptive Segmentation...")
    df['segments'] = df.progress_apply(
        lambda x: adaptive_segmentation_2(x['sentences'], x['embeddings'], k=args.k_threshold), axis=1
    )

    df[['seg_sentences', 'seg_embeddings', 'seg_len']] = df.progress_apply(
        lambda x: pd.Series(prepare_segments_for_tree_2(x['sentences'], x['embeddings'], x['segments'])), axis=1
    )

    print("Building Trees...")
    df['tree'] = df.progress_apply(
        lambda x: recursive_tree_build(x['seg_sentences'], x['seg_embeddings'], x['book_id'], max_depth=args.max_depth), axis=1
    )

    print("Core Identifier...")
    df['conflict'] = df.progress_apply(lambda x: 
        identify_conflict(x['book_id'], x['leaf_texts'])
    , axis = 1)

    print("Event Selector...")
    df['leaf_texts2'] = df.progress_apply(lambda x: 
        rank_nodes(x['book_id'], x['leaf_texts'], x['conflict'], target_summary_tokens = x['summary_length'])
    , axis = 1)

    print("Final Summarization...")
    df['polished'] = df.progress_apply(lambda x: 
        polish1(x.name, x['book_id'], x['leaf_texts2'], target_words=x['summary_length'], style=x['source']) 
    , axis = 1)

    print("Evaluation...")
    df[['P', 'R', 'F1']] = df.progress_apply(lambda x: 
        compute_bertscore(x['polished'], x['summary']) if len(x['polished'])>0 else [None] * 8
    , axis = 1, result_type='expand')

    df[['rouge1', 'rouge2', 'rougeL', 'geometric_mean']] = df.progress_apply(lambda x: 
        compute_rouge(x['polished'], x['summary']) if len(x['polished'])>0 else [None] * 8
    , axis = 1, result_type='expand')

    df.to_csv('results.csv', index=False)
    print("Results are saved")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full Text-to-MindMap Summarization Pipeline")
    parser.add_argument("--dataset", type=str, default="ubaada/booksum-complete-cleaned")
    parser.add_argument("--embed_model", type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--model", type=str, default="mistral-small3.2:24b")
    parser.add_argument("--max_depth", type=int, default=2)
    parser.add_argument("--k_threshold", type=float, default=0.9)
    args = parser.parse_args()
    main(args)
