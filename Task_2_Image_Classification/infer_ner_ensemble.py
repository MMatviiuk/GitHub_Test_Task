"""
Performs ensemble NER inference with two different models (BERT + RoBERTa).
Merges results via union for a more flexible match.
"""

import argparse
import os
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline
)
import torch

def parse_args():
    """
    Parses command-line arguments for the ensemble NER inference.
    """
    parser = argparse.ArgumentParser(description="Ensemble NER inference using two models (BERT + RoBERTa).")
    parser.add_argument("--text", type=str, required=True, help="Input text to analyze.")
    parser.add_argument("--bert_model_dir", type=str, required=True, help="Directory of the BERT model.")
    parser.add_argument("--roberta_model_dir", type=str, required=True, help="Directory of the RoBERTa model.")
    return parser.parse_args()

def clean_word(word: str) -> str:
    """
    Removes RoBERTa-specific artifacts like 'Ġ' and leading/trailing spaces.
    """
    return word.replace("Ġ", "").strip()

def extract_entities(text, model_dir):
    """
    Uses a Transformers pipeline with 'aggregation_strategy'='simple'
    to extract entities from the input text.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)

    ner_pipe = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple"
    )

    results = ner_pipe(text)
    entities = [clean_word(r["word"]) for r in results]

    return entities

def combine_entities(bert_ents, roberta_ents):
    """
    Combine results from both models using union for a more flexible match.
    """
    set_bert = set(ent.lower() for ent in bert_ents)
    set_roberta = set(ent.lower() for ent in roberta_ents)
    final = sorted(set_bert.union(set_roberta))  # Используем объединение вместо пересечения
    return final

def main():
    args = parse_args()

    print("[STEP] Extracting with BERT model...")
    bert_entities = extract_entities(args.text, args.bert_model_dir)
    print(f"[INFO] BERT Entities: {bert_entities}")

    print("[STEP] Extracting with RoBERTa model...")
    roberta_entities = extract_entities(args.text, args.roberta_model_dir)
    print(f"[INFO] RoBERTa Entities: {roberta_entities}")

    print("[STEP] Combining results...")
    final_ents = combine_entities(bert_entities, roberta_entities)
    print(f"[INFO] Final Ensemble Entities: {final_ents}")

if __name__ == "__main__":
    main()
