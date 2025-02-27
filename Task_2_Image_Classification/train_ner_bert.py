"""
Trains a BERT-based NER model for detecting animal names in text.
Automatically generates synthetic NER data if the specified JSON files do not exist.
Uses a set of animals mapped from Italian to English.
"""

import argparse
import os
import json
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer
)

# Label scheme for the "ANIMAL" entity using BIO tagging
LABEL2ID = {"O": 0, "B-ANIMAL": 1, "I-ANIMAL": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# Mapping from Italian animals to English animals
ITALIAN_TO_ENGLISH = {
    "cane": "dog",
    "cavallo": "horse",
    "elefante": "elephant",
    "farfalla": "butterfly",
    "gallina": "chicken",
    "gatto": "cat",
    "mucca": "cow",
    "pecora": "sheep",
    "ragno": "spider",
    "scoiattolo": "squirrel"
}

# Sentence templates to create synthetic data
SENTENCE_TEMPLATES = [
    "There is a {animal} in the picture.",
    "I saw a {animal} near the lake.",
    "A {animal} is walking in the park.",
    "The {animal} runs in the field.",
    "Look at the {animal} next to the tree.",
    "Have you seen a {animal} here?",
    "A {animal} was found in the garden.",
    "The photo shows a {animal}.",
    "Someone mentioned a {animal} on the roof.",
    "Is that a {animal} behind the fence?"
]

def parse_args():
    """
    Parse command-line arguments for training the BERT-based NER model.
    """
    parser = argparse.ArgumentParser(description="Train a BERT-based NER model for animal detection.")
    parser.add_argument("--train_file", type=str, default="./ner_data/ner_train.json")
    parser.add_argument("--val_file", type=str, default="./ner_data/ner_val.json")
    parser.add_argument("--model_name", type=str, default="bert-base-cased")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default="./ner_model_bert")
    parser.add_argument("--no_cuda", action='store_true', help="Disable CUDA training.")
    return parser.parse_args()

def maybe_generate_json(train_path, val_path):
    """
    Automatically generate synthetic NER data if the specified JSON files do not exist.
    """
    if not os.path.exists(os.path.dirname(train_path)):
        os.makedirs(os.path.dirname(train_path), exist_ok=True)

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        synthetic_data = []
        for eng_animal in ITALIAN_TO_ENGLISH.values():
            for template in SENTENCE_TEMPLATES:
                sentence = template.format(animal=eng_animal)
                start_idx = sentence.lower().find(eng_animal.lower())
                end_idx = start_idx + len(eng_animal)
                synthetic_data.append({
                    "text": sentence,
                    "entities": [{"start": start_idx, "end": end_idx, "label": "ANIMAL"}]
                })

        train_size = int(len(synthetic_data) * 0.8)
        with open(train_path, "w", encoding="utf-8") as f_train:
            json.dump(synthetic_data[:train_size], f_train, indent=4)
        with open(val_path, "w", encoding="utf-8") as f_val:
            json.dump(synthetic_data[train_size:], f_val, indent=4)
        print(f"[INFO] Synthetic NER data created at: {train_path} and {val_path}")

def tokenize_and_align_labels(examples, tokenizer, max_length):
    """
    Tokenize text and align B/I/O labels accordingly to each token.
    """
    tokenized_inputs = tokenizer(
        examples["text"],
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_offsets_mapping=True
    )
    all_labels = []
    for i, offsets in enumerate(tokenized_inputs["offset_mapping"]):
        char_labels = ["O"] * len(examples["text"][i])
        for ent in examples["entities"][i]:
            for pos in range(ent["start"], ent["end"]):
                char_labels[pos] = "B-ANIMAL" if pos == ent["start"] else "I-ANIMAL"
        subword_labels = [LABEL2ID.get(char_labels[start], 0) if start < len(char_labels) else -100
                           for start, end in offsets]
        all_labels.append(subword_labels)
    tokenized_inputs["labels"] = all_labels
    tokenized_inputs.pop("offset_mapping", None)
    return tokenized_inputs

def main():
    """
    Main training pipeline for the BERT-based NER model.
    """
    args = parse_args()
    os.environ["WANDB_DISABLED"] = "true"
    maybe_generate_json(args.train_file, args.val_file)

    with open(args.train_file, "r", encoding="utf-8") as f_train:
        train_data = json.load(f_train)
    with open(args.val_file, "r", encoding="utf-8") as f_val:
        val_data = json.load(f_val)

    dataset = DatasetDict({
        "train": Dataset.from_list(train_data),
        "validation": Dataset.from_list(val_data)
    })

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    tokenized_dataset = dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, args.max_length),
        batched=True
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to="none",
        no_cuda=args.no_cuda  # Prevent CUDA errors if GPU issues occur
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"]
    )

    trainer.train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[INFO] BERT model training complete. Saved at '{args.output_dir}'.")

if __name__ == "__main__":
    main()
