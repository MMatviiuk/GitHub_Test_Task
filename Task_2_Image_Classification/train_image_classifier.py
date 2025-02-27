"""
Trains a ResNet18 image classification model on a set of animal classes.
The dataset folder has subdirectories for each class.
"""

import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from tqdm import tqdm  # Added for progress display

def parse_args():
    """
    Parses command-line arguments for training the image classification model.
    """
    parser = argparse.ArgumentParser(description="Train a ResNet18-based image classification model for multiple animal classes.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/kaggle/input/animals10/raw-img/",  
        help="Path to the dataset directory containing subfolders for each animal class."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate."
    )
    parser.add_argument(
        "--output_model",
        type=str,
        default="./image_classifier.pth",
        help="Path to save the trained model's state_dict."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"[ERROR] Data directory '{args.data_dir}' not found. Check the Kaggle input path.")

    # Image transformation pipeline
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(root=args.data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Load pretrained ResNet18 model using updated 'weights' argument
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    num_classes = len(dataset.classes)
    print(f"[INFO] Number of classes: {num_classes}, Classes: {dataset.classes}")  
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9) 

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()

    for epoch in range(args.epochs):
        total_loss = 0.0
        print(f"\n[INFO] Starting epoch {epoch+1}/{args.epochs}...")
        
        # Added tqdm for progress bar
        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"[INFO] Epoch {epoch+1}/{args.epochs} complete. Average Loss: {avg_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), args.output_model)
    print(f"[INFO] Model training complete. Saved at: {args.output_model}")

if __name__ == "__main__":
    main()

