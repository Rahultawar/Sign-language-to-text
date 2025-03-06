import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import mediapipe as mp
from utils import preprocess_image, load_data  

def load_trained_model(model_dir):
    # Load model architecture
    with open(f"{model_dir}/model_architecture.json", "r") as f:
        model_architecture = f.read()
    model = tf.keras.models.model_from_json(model_architecture)
    
    # Load model weights
    model.load_weights(f"{model_dir}/trained_model.h5")
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def test_model(model, data_dir, scaler):
    X_test, y_test = load_data(data_dir, scaler)
    
    # Convert labels to one-hot encoding
    y_test = pd.get_dummies(y_test).values
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_classes = y_pred.argmax(axis=1)
    y_true_classes = y_test.argmax(axis=1)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_true_classes, y_pred_classes)
    print(f"Test Accuracy: {accuracy}")
    
    # Print classification report
    print(classification_report(y_true_classes, y_pred_classes))

if __name__ == "__main__":
    model_dir = "models"
    test_data_dir = "data/test"  # Directory containing test images
    scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
    model = load_trained_model(model_dir)
    test_model(model, test_data_dir, scaler)