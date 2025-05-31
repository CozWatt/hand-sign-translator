import os
import random
import shutil
from sklearn.model_selection import train_test_split

random.seed(42)  # For reproducibility

processed_path = "C:/HandSignTranslator/processed_dataset"
train_path = "C:/HandSignTranslator/train"
test_path = "C:/HandSignTranslator/test"

# Create directories
os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

for category in os.listdir(processed_path):
    category_path = os.path.join(processed_path, category)
    if not os.path.isdir(category_path):
        continue
    
    # Get all files and split
    files = [f for f in os.listdir(category_path) if f.endswith('.npy')]
    if not files:
        print(f"No .npy files found in {category_path}")
        continue
    
    train_files, test_files = train_test_split(
        files, test_size=0.2, random_state=42
    )
    
    # Create category directories
    os.makedirs(os.path.join(train_path, category), exist_ok=True)
    os.makedirs(os.path.join(test_path, category), exist_ok=True)
    
    # Copy files
    for f in train_files:
        src = os.path.join(category_path, f)
        dst = os.path.join(train_path, category, f)
        shutil.copy(src, dst)
    
    for f in test_files:
        src = os.path.join(category_path, f)
        dst = os.path.join(test_path, category, f)
        shutil.copy(src, dst)
    
    print(f"{category}: {len(train_files)} train, {len(test_files)} test")

print("Dataset split completed!")