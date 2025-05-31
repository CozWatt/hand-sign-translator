import cv2
import os
import numpy as np

dataset_path = r"C:\HandSignTranslator\dataset"
processed_path = r"C:\HandSignTranslator\processed_dataset"

# Create directories if they don't exist
os.makedirs(processed_path, exist_ok=True)

for category in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, category)
    save_path = os.path.join(processed_path, category)
    
    if not os.path.isdir(class_path):
        continue
        
    os.makedirs(save_path, exist_ok=True)
    
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Failed to load {img_path}")
                
            img = cv2.resize(img, (64, 64))
            img = img.astype('float32') / 255.0  # Ensure float32 dtype
            np.save(os.path.join(save_path, os.path.splitext(img_name)[0] + ".npy"), img)
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            continue

print("Preprocessing completed successfully!")