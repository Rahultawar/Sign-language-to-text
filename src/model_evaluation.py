import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import mediapipe as mp
from utils import preprocess_image, load_data  

def evaluate_model(model, X_val, y_val, labels):
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_val, axis=1)

    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

if __name__ == "__main__":
    # Load the trained model
    model = load_model('models/trained_model.h5')

    # Load the scaler
    scaler = joblib.load('models/scaler.pkl')

    # Load the validation data
    val_data_dir = "data/val"
    X_val, y_val = load_data(val_data_dir, scaler)

    # Convert labels to one-hot encoding
    y_val = pd.get_dummies(y_val).values

    # Define the labels
    labels = ['hello', 'my', 'name', 'is', 'rahul']

    # Evaluate the model
    evaluate_model(model, X_val, y_val, labels)