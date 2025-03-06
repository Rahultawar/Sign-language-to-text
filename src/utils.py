import os
import cv2
import numpy as np
import mediapipe as mp
import joblib

# Initialize MediaPipe for hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

def preprocess_image(image_path, scaler):
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
            features = np.array(features).reshape(1, -1)
            normalized_features = scaler.transform(features)
            return normalized_features
    return None

def load_data(data_dir, scaler):
    data = []
    labels = []
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            for filename in os.listdir(label_dir):
                if filename.endswith('.png'):
                    image_path = os.path.join(label_dir, filename)
                    features = preprocess_image(image_path, scaler)
                    if features is not None:
                        data.append(features)
                        labels.append(label)
    data = np.array(data)
    labels = np.array(labels)
    # Ensure the data has the correct shape
    data = data.reshape(data.shape[0], data.shape[2], 1)
    return data, labels