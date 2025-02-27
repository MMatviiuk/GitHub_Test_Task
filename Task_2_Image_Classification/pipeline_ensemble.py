"""
Pipeline that combines text-based NER (BERT + RoBERTa ensemble) 
and an image classifier to verify if the animal mentioned in the text matches the image.
- Displays analyzed images (original and resized 100x100 px).
- Provides a final boolean output indicating consistency.
- Handles file errors with informative messages and ensures final image display.
"""

import argparse
import torch
import os
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt
from IPython.display import display
import torchvision.transforms as transforms
from torchvision import models

from infer_ner_ensemble import extract_entities, combine_entities

# Import translation dictionary
import sys
sys.path.insert(0, '/kaggle/input/animals10/')
try:
    from translate import translate
except ImportError:
    raise ImportError("[ERROR] translate.py not found in /kaggle/input/animals10/.")

def parse_args():
    """
    Parse command-line arguments for pipeline execution.
    """
    parser = argparse.ArgumentParser(description="Pipeline: NER (Ensemble) + Image Classification.")
    parser.add_argument("--text", type=str, required=True, help="Text containing an animal mention.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--bert_model_dir", type=str, required=True, help="Path to the BERT model directory.")
    parser.add_argument("--roberta_model_dir", type=str, required=True, help="Path to the RoBERTa model directory.")
    parser.add_argument("--img_model", type=str, default="./image_classifier.pth", help="Path to the image classifier model.")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of animal classes.")
    return parser.parse_args()

def load_image(file_path):
    """
    Load and verify an image. Return None if invalid.
    """
    try:
        img = Image.open(file_path)
        img.verify()
        img = Image.open(file_path)
        img = img.convert("RGB")
        return img
    except (UnidentifiedImageError, OSError, AttributeError) as e:
        print(f"[WARNING] Skipped corrupted file: {file_path} ({e})")
        return None

def display_image_with_small(img, title="Analyzed Image"):
    """
    Display the original and a 100x100 px version of the image using Matplotlib.

    Args:
        img (PIL.Image.Image): Image to display.
        title (str): Title for the original image.
    """
    # Display original image
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.title(title)
    plt.show()

    # Display small image
    print("[STEP] Displaying small sample image (100x100 px)...")
    small_img = img.resize((100, 100))
    plt.figure(figsize=(2, 2))
    plt.imshow(small_img)
    plt.axis('off')
    plt.title("Small Image (100x100 px)")
    plt.show()
    print("[INFO] Small sample image displayed successfully.")

def classify_image(image_path, model_path, num_classes):
    """
    Classify the input image using a ResNet18 model.

    Returns:
        int: Predicted class index or None if classification fails.
    """
    try:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        image = load_image(image_path)
        if image is None:
            return None

        image_tensor = transform(image).unsqueeze(0)

        model = models.resnet18()
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        model.eval()

        with torch.no_grad():
            outputs = model(image_tensor)
            pred_idx = torch.argmax(outputs, dim=1).item()

        return pred_idx
    except FileNotFoundError:
        print(f"[ERROR] Image not found at: {image_path}. Classification aborted.")
        return None

def display_final_image(img, predicted_animal, is_correct):
    """
    Display the final 100x100 px image with match result.

    Args:
        img (PIL.Image.Image): Image to display.
        predicted_animal (str): Predicted animal name.
        is_correct (bool): Whether text and image match.
    """
    print("[STEP] Displaying final result image (100x100 px)...")
    small_img = img.resize((100, 100))
    plt.figure(figsize=(2, 2))
    plt.imshow(small_img)
    plt.axis('off')
    plt.title(f"Final Result: {predicted_animal} (Match: {is_correct})")
    plt.savefig('/kaggle/working/final_image.png')  # Save to file
    plt.close()  # Close current figure to avoid overlap
    from IPython.display import Image as IPImage
    display(IPImage('/kaggle/working/final_image.png'))  # Explicit display
    print("[INFO] Final result image displayed successfully.")

def main():
    """
    Main pipeline combining NER extraction and image classification.
    Ensures final small image display after result.
    """
    args = parse_args()

    # Check image path
    if not os.path.exists(args.image_path):
        print(f"[ERROR] The specified image path does not exist: {args.image_path}")
        return

    # Map classes from Italian to English
    italian_classes = ["cane", "cavallo", "elefante", "farfalla", "gallina", "gatto", "mucca", "pecora", "ragno", "scoiattolo"]
    english_classes = [translate.get(it_class, it_class) for it_class in italian_classes]

    # Load and display initial image
    print("[STEP] Displaying the analyzed image and small version...")
    img = load_image(args.image_path)
    if img:
        display_image_with_small(img)
        print(f"[DEBUG] Image size: {img.size}, mode: {img.mode}")
    else:
        print("[ERROR] Image loading failed.")
        return

    # NER extraction
    print("[STEP] Extracting entities (NER) from text...")
    bert_ents = extract_entities(args.text, args.bert_model_dir)
    roberta_ents = extract_entities(args.text, args.roberta_model_dir)
    final_ents = combine_entities(bert_ents, roberta_ents)
    print(f"[INFO] Recognized entities: {final_ents}")

    if not final_ents:
        print("[RESULT] False (No entity recognized in the text)")
        display_final_image(img, "None", False)
        return

    claimed_animal = final_ents[0].lower()

    # Image classification
    print("[STEP] Classifying the provided image...")
    predicted_idx = classify_image(args.image_path, args.img_model, args.num_classes)
    if predicted_idx is None:
        return

    predicted_animal = english_classes[predicted_idx] if predicted_idx < len(english_classes) else str(predicted_idx)
    print(f"[INFO] Predicted animal (English): {predicted_animal}")

    # Display classified image and small version
    print("[STEP] Displaying image after classification...")
    display_image_with_small(img, title=f"Predicted: {predicted_animal}")

    # Final result with guaranteed small image display
    is_correct = (claimed_animal == predicted_animal.lower())
    print(f"[FINAL RESULT] Text and image match: {is_correct}")
    display_final_image(img, predicted_animal, is_correct)

if __name__ == "__main__":
    main()