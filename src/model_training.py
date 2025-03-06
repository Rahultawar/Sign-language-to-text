import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import mediapipe as mp

# Initialize MediaPipe for hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            features = []
            for landmark in hand_landmarks.landmark:
                features.append(landmark.x)
                features.append(landmark.y)
                features.append(landmark.z)
            return np.array(features)
    return None

def load_data(data_dir):
    data = []
    labels = []
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            for filename in os.listdir(label_dir):
                if filename.endswith('.png'):
                    image_path = os.path.join(label_dir, filename)
                    features = preprocess_image(image_path)
                    if features is not None:
                        data.append(features)
                        labels.append(label)
    return np.array(data), np.array(labels)

def train_model(train_dir, val_dir, model_dir):
    # Load and preprocess the data
    X_train, y_train = load_data(train_dir)
    X_val, y_val = load_data(val_dir)
    
    # Normalize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    # Save the scaler
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    
    # Convert labels to one-hot encoding
    y_train = pd.get_dummies(y_train).values
    y_val = pd.get_dummies(y_val).values
    
    # Reshape X for LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)  
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    
    # Define model architecture (LSTM used for sequence data or time series)
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(y_train.shape[1], activation='softmax')
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Train the model
    model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_val, y_val))
    
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model weights
    model.save(os.path.join(model_dir, 'trained_model.h5'))
    print(f"Model weights saved to {model_dir}/trained_model.h5")
    
    # Save model architecture
    with open(os.path.join(model_dir, 'model_architecture.json'), "w") as f:
        f.write(model.to_json())
    print(f"Model architecture saved to {model_dir}/model_architecture.json")

if __name__ == "__main__":
    train_dir = "data/train"
    val_dir = "data/val"
    model_dir = "models"
    train_model(train_dir, val_dir, model_dir)