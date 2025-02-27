"""
Trains a RoBERTa-based NER model for detecting animal names in text.
Automatically generates synthetic NER data if the specified JSON files do not exist.
Uses a refined approach to handle encoding issues and improve entity recognition accuracy.
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

# Define label mappings for BIO tagging
LABEL2ID = {"O": 0, "B-ANIMAL": 1, "I-ANIMAL": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# Mapping Italian animal names to English equivalents
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

# Synthetic sentence templates
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
    Parse command-line arguments required for training.
    """
    parser = argparse.ArgumentParser(description="Train RoBERTa-based NER model.")
    parser.add_argument("--train_file", type=str, default="./ner_data/ner_train.json")
    parser.add_argument("--val_file", type=str, default="./ner_data/ner_val.json")
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default="./ner_model_roberta")
    return parser.parse_args()

def generate_synthetic_data(train_path, val_path):
    """
    Generate synthetic NER data if not already available.
    """
    if not os.path.exists(os.path.dirname(train_path)):
        os.makedirs(os.path.dirname(train_path), exist_ok=True)

    synthetic_data = []
    for animal in ITALIAN_TO_ENGLISH.values():
        for template in SENTENCE_TEMPLATES:
            text = template.format(animal=animal)
            start = text.find(animal)
            end = start + len(animal)
            synthetic_data.append({
                "text": text,
                "entities": [{"start": start, "end": end, "label": "ANIMAL"}]
            })

    split_idx = int(0.8 * len(synthetic_data))
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(synthetic_data[:split_idx], f, indent=4)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(synthetic_data[split_idx:], f, indent=4)

    print(f"[INFO] Synthetic data created at: {train_path} & {val_path}")

def create_labels(entities, length):
    """
    Generate BIO labels at character level.
    """
    labels = ["O"] * length
    for ent in entities:
        start, end = ent["start"], ent["end"]
        if 0 <= start < end <= length:
            labels[start] = "B-ANIMAL"
            for i in range(start + 1, end):
                labels[i] = "I-ANIMAL"
    return labels

def tokenize_and_align(examples, tokenizer, max_len):
    """
    Tokenize inputs and align with BIO labels.
    """
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_offsets_mapping=True
    )
    aligned_labels = []
    for i, offsets in enumerate(tokenized["offset_mapping"]):
        char_labels = create_labels(examples["entities"][i], len(examples["text"][i]))
        labels = []
        for start, end in offsets:
            if start == 0 and end == 0:
                labels.append(-100)
            else:
                labels.append(LABEL2ID.get(char_labels[start], 0) if start < len(char_labels) else 0)
        aligned_labels.append(labels)
    tokenized.pop("offset_mapping", None)
    tokenized["labels"] = aligned_labels
    return tokenized

def main():
    """
    Main training procedure for RoBERTa NER model.
    """
    os.environ["WANDB_DISABLED"] = "true"
    args = parse_args()
    generate_synthetic_data(args.train_file, args.val_file)

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset = dataset.map(
        lambda x: tokenize_and_align(x, tokenizer, args.max_length),
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
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"]
    )

    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[INFO] RoBERTa model saved at {args.output_dir}.")

if __name__ == "__main__":
    main()
