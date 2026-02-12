import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys

# --- CONFIGURATION ---
MODEL_PATH = 'dental_detective_model.pth'
TEST_DIR = 'dataset_split/test'
TRAIN_DIR = 'dataset_split/train'
BATCH_SIZE = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(directory):
    """Loads data with standard normalization"""
    # Note: Ensure these transforms match what you used during training!
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    data = datasets.ImageFolder(directory, transform=test_transforms)
    loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=False)
    return loader, data.classes

def get_predictions(model, loader):
    """Runs inference and returns true labels vs predicted labels"""
    model.eval()
    all_preds = []
    all_labels = []
    
    print(f"Running inference on {len(loader.dataset)} images...")
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    return np.array(all_labels), np.array(all_preds)

def main():
    # 1. Load Model
    print(f"Loading model from {MODEL_PATH}...")
    try:
        # Try loading the full model (if saved with torch.save(model))
        model = torch.load(MODEL_PATH, map_location=device)
    except Exception as e:
        print(f"⚠️ Error loading full model: {e}")
        print("Attempting to load state_dict... (You might need to import your model class here!)")
        # If you have a specific class in app.py, import it: from app import MyModel
        # model = MyModel() 
        # model.load_state_dict(torch.load(MODEL_PATH))
        return # Stop if we can't load the model

    model.to(device)
    
    # 2. Prepare Data
    if not os.path.exists(TEST_DIR):
        print(f"❌ Error: Test directory '{TEST_DIR}' not found.")
        return

    test_loader, class_names = load_data(TEST_DIR)
    
    # 3. Get Predictions
    y_true, y_pred = get_predictions(model, test_loader)
    
    # 4. Generate Reports
    print("\n" + "="*30)
    print("CLASSIFICATION REPORT")
    print("="*30)
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Weighted F1 Score: {f1:.4f}")

    # 5. Plot Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # 6. Check for Overfitting/Underfitting (Optional Check on Train Data)
    print("\n" + "="*30)
    print("OVERFIT/UNDERFIT CHECK")
    print("="*30)
    
    # Calculate Test Accuracy again (we already have it)
    test_acc = acc
    
    # Calculate Train Accuracy (using a subset to save time)
    train_loader, _ = load_data(TRAIN_DIR)
    # We take just the first 10 batches to get a quick estimate
    limit_batches = 10
    train_correct = 0
    train_total = 0
    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(train_loader):
            if i >= limit_batches: break
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (preds == labels.to(device)).sum().item()
            
    train_acc = train_correct / train_total
    
    print(f"Training Accuracy (est.): {train_acc:.4f}")
    print(f"Testing Accuracy:        {test_acc:.4f}")
    
    if train_acc > 0.95 and test_acc < 0.80:
        print("🔴 Diagnosis: OVERFITTING (Model memorized training data but fails on new data)")
        print("   -> Fix: Add Dropout, increase data augmentation, or use L2 regularization.")
    elif train_acc < 0.70 and test_acc < 0.70:
        print("🟡 Diagnosis: UNDERFITTING (Model is too simple to learn the patterns)")
        print("   -> Fix: Use a deeper model (e.g., ResNet50 instead of 18) or train for more epochs.")
    else:
        print("🟢 Diagnosis: GOOD FIT (Train and Test scores are close and high)")

if __name__ == "__main__":
    main()