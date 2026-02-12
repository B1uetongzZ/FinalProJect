import os
import shutil
import random
import cv2
import numpy as np
from pathlib import Path

# ================= CONFIGURATION =================
# Path to your source folder (where 'calculus', 'discoloration', 'ulcer' are)
INPUT_DIR = "dataset_raw" 

# Where you want the output to go
OUTPUT_DIR = "dataset_split"

# Classes to process (matches your folder names)
CLASSES = ['calculus', 'discoloration', 'ulcer']

# Split Ratios (Must add up to 1.0)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Augmentation Settings (How many new images per original image?)
AUGMENTATIONS_PER_IMAGE = 2 
# =================================================

def adjust_lighting(image):
    """Simulate 'Good/Varied Lighting' by randomly adjusting brightness/contrast"""
    alpha = random.uniform(0.8, 1.2) # Contrast control
    beta = random.randint(-30, 30)   # Brightness control
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted

def adjust_angle(image):
    """Simulate 'Good Angles' using rotation and flipping"""
    # 1. Random Flip (Horizontal) - 50% chance
    if random.random() > 0.5:
        image = cv2.flip(image, 1)

    # 2. Random Rotation (between -15 and +15 degrees)
    # We keep rotation slight to preserve the dental context
    angle = random.uniform(-15, 15)
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    return rotated

def augment_image(image):
    """Apply both lighting and angle adjustments"""
    img = adjust_angle(image)
    img = adjust_lighting(img)
    return img

def create_dir_structure():
    """Creates the train/val/test folders"""
    for split in ['train', 'val', 'test']:
        for cls in CLASSES:
            os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

def process_dataset():
    create_dir_structure()
    
    for cls in CLASSES:
        source_path = os.path.join(INPUT_DIR, cls)
        if not os.path.exists(source_path):
            print(f"Skipping {cls} (folder not found)")
            continue

        # Get list of all images
        all_files = [f for f in os.listdir(source_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(all_files) # Shuffle to ensure random split
        
        total_files = len(all_files)
        train_count = int(total_files * TRAIN_RATIO)
        val_count = int(total_files * VAL_RATIO)
        
        # Slicing the lists
        train_files = all_files[:train_count]
        val_files = all_files[train_count : train_count + val_count]
        test_files = all_files[train_count + val_count :]

        print(f"\nProcessing Class: {cls}")
        print(f"  - Total: {total_files}")
        print(f"  - Train: {len(train_files)} (will be augmented)")
        print(f"  - Val:   {len(val_files)}")
        print(f"  - Test:  {len(test_files)}")

        # --- PROCESS VALIDATION & TEST (Just Copy) ---
        for f in val_files:
            shutil.copy2(os.path.join(source_path, f), os.path.join(OUTPUT_DIR, 'val', cls, f))
        for f in test_files:
            shutil.copy2(os.path.join(source_path, f), os.path.join(OUTPUT_DIR, 'test', cls, f))

        # --- PROCESS TRAINING (Copy + Augment) ---
        for f in train_files:
            img_path = os.path.join(source_path, f)
            original_img = cv2.imread(img_path)
            
            if original_img is None:
                continue

            # 1. Save Original
            cv2.imwrite(os.path.join(OUTPUT_DIR, 'train', cls, f), original_img)

            # 2. Generate Augmented Versions
            base_name = os.path.splitext(f)[0]
            ext = os.path.splitext(f)[1]
            
            for i in range(AUGMENTATIONS_PER_IMAGE):
                aug_img = augment_image(original_img)
                new_name = f"{base_name}_aug_{i}{ext}"
                cv2.imwrite(os.path.join(OUTPUT_DIR, 'train', cls, new_name), aug_img)

    print(f"\n✅ Done! Dataset saved to '{OUTPUT_DIR}'")

if __name__ == "__main__":
    process_dataset()