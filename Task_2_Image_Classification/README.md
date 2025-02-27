# Task 2: Named Entity Recognition and Image Classification Pipeline

## Overview

This project implements a complete machine learning pipeline combining two distinct tasks:

1. **Named Entity Recognition (NER):** Using transformer-based models (BERT and RoBERTa) to extract animal entities from user-provided text.
2. **Image Classification:** Utilizing a ResNet18-based model to classify animal images into ten predefined categories.

The final pipeline matches the extracted animal entity from the text with the predicted animal class from the image, outputting a boolean value indicating the consistency of the match.

---

## Project Structure

```
project_root/
│
├── EDA.ipynb                    # Exploratory Data Analysis notebook
├── infer_ner_ensemble.py        # NER ensemble inference (BERT + RoBERTa)
├── pipeline_ensemble.py         # Final pipeline combining NER and image classification
├── requirements.txt             # Project dependencies
├── train_image_classifier.py    # Training script for ResNet18 image classifier
├── train_ner_bert.py            # BERT NER training script
├── train_ner_roberta.py         # RoBERTa NER training script
├── check.ipynb                  # Final validation notebook for Kaggle
│
├── ner_data/                    # Synthetic NER training/validation data
│   ├── ner_train.json
│   └── ner_val.json
│
├── ner_model_bert/              # Saved BERT model and tokenizer
├── ner_model_roberta/           # Saved RoBERTa model and tokenizer
│
└── image_classifier.pth         # Trained image classification model
```

---

## Installation

Ensure all dependencies are installed:

```bash
!pip install -r requirements.txt --quiet
```

---

## Exploratory Data Analysis (EDA)

Run the EDA notebook to:
- Verify dataset integrity.
- Visualize image samples per category.
- Display data distribution.

```bash
!jupyter nbconvert --to notebook --execute --inplace EDA.ipynb
%run EDA.ipynb
```

---

## Model Training

### Image Classifier Training

```bash
!python train_image_classifier.py \
    --data_dir /kaggle/input/animals10/raw-img/ \
    --epochs 10 \
    --batch_size 32 \
    --lr 0.001 \
    --output_model ./image_classifier.pth
```

### BERT NER Model Training

```bash
!python train_ner_bert.py \
    --train_file ./ner_data/ner_train.json \
    --val_file ./ner_data/ner_val.json \
    --model_name bert-base-cased \
    --epochs 10 \
    --batch_size 16 \
    --lr 3e-5 \
    --max_length 64 \
    --output_dir ./ner_model_bert
```

### RoBERTa NER Model Training

```bash
!python train_ner_roberta.py \
    --train_file ./ner_data/ner_train.json \
    --val_file ./ner_data/ner_val.json \
    --model_name roberta-base \
    --epochs 10 \
    --batch_size 16 \
    --lr 3e-5 \
    --max_length 64 \
    --output_dir ./ner_model_roberta
```

---

## Ensemble Inference Example

```bash
!python infer_ner_ensemble.py \
    --text "I saw a cat near the lake." \
    --bert_model_dir ./ner_model_bert \
    --roberta_model_dir ./ner_model_roberta
```

**Expected Output:**
```
[STEP] Extracting with BERT model...
[INFO] BERT Entities: ['cat']
[STEP] Extracting with RoBERTa model...
[INFO] RoBERTa Entities: ['cat']
[STEP] Combining results...
[INFO] Final Ensemble Entities: ['cat']
```

---

## Full Pipeline Test

Run the complete pipeline with a given text and image:

```bash
!python pipeline_ensemble.py \
    --text "There is an elephant in the picture." \
    --image_path /kaggle/input/animals10/raw-img/elefante/OIP--NEqn4JVnn251xGu7ss4bQHaHa.jpeg \
    --bert_model_dir ./ner_model_bert \
    --roberta_model_dir ./ner_model_roberta \
    --img_model ./image_classifier.pth \
    --num_classes 10
```

**Expected Output:**
```
[STEP] Displaying the analyzed image...
Figure(600x600)
[STEP] Extracting entities (NER) from user-provided text...
[INFO] Recognized entities: ['elephant']
[STEP] Classifying the provided image...
[INFO] Predicted animal (English): elephant
Figure(600x600)
[FINAL RESULT] Text and image match: True
```

---

## Kaggle Verification (`check.ipynb`)

The `check.ipynb` notebook runs all scripts in the correct sequence to verify project consistency. Execute it directly on Kaggle:

```bash
!jupyter nbconvert --to notebook --execute --inplace check.ipynb
```

The output cells provide full logs confirming the operational status of each step.

---

## Notes

- Ensure GPU runtime is enabled on Kaggle for efficient training.
- The `check.ipynb` notebook serves as the final proof of pipeline functionality.
- Adjust hyperparameters (`epochs`, `batch_size`, `lr`) as needed based on resource availability and performance goals.
