import os
import shutil
from sklearn.model_selection import train_test_split

def split_data(input_dir, train_dir, val_dir, test_dir, test_size=0.2, val_size=0.1):
    # Create directories for saving the data if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for label in os.listdir(input_dir):
        label_dir = os.path.join(input_dir, label)
        if os.path.isdir(label_dir):
            images = [os.path.join(label_dir, img) for img in os.listdir(label_dir) if img.endswith('.png')]
            
            # Split into training, validation, and test sets
            train_images, temp_images = train_test_split(images, test_size=test_size + val_size, random_state=42)
            val_images, test_images = train_test_split(temp_images, test_size=test_size / (test_size + val_size), random_state=42)
            
            # Create label directories in train, val, and test directories
            os.makedirs(os.path.join(train_dir, label), exist_ok=True)
            os.makedirs(os.path.join(val_dir, label), exist_ok=True)
            os.makedirs(os.path.join(test_dir, label), exist_ok=True)
            
            # Move images to respective directories
            for img in train_images:
                shutil.copy(img, os.path.join(train_dir, label))
            for img in val_images:
                shutil.copy(img, os.path.join(val_dir, label))
            for img in test_images:
                shutil.copy(img, os.path.join(test_dir, label))

    print("Data split into train, validation, and test sets and saved successfully.")

if __name__ == "__main__":
    input_dir = "data/raw"
    train_dir = "data/train"
    val_dir = "data/val"
    test_dir = "data/test"
    split_data(input_dir, train_dir, val_dir, test_dir)