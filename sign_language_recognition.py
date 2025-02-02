import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp

# Load the trained model
model = load_model('models/trained_model.h5')
model.summary()

# Recompile the model to suppress the warning
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Initialize MediaPipe for hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

def preprocess_frame(frame, landmarks):
    # Extract hand landmarks and normalize
    features = []
    for landmark in landmarks:
        features.append(landmark.x)
        features.append(landmark.y)
        features.append(landmark.z)
    features = np.array(features)
    normalized_frame = features / np.max(features)  
    reshaped_frame = np.reshape(normalized_frame, (1, len(features), 1))  # Adjust based on features
    return reshaped_frame

def get_label(prediction):
    labels = ['hello', 'my', 'name', 'is', 'rahul']  
    predicted_index = np.argmax(prediction)
    
    # Debugging
    print(f"Prediction: {prediction}")  # Print the model's output
    print(f"Predicted Index: {predicted_index}")  # Print the predicted index
    print(f"Number of Labels: {len(labels)}")  # Print the length of the labels list
    
    # Check if the predicted index is valid
    if predicted_index >= len(labels):
        print("Error: predicted index is out of bounds.")
        return "Unknown"  # Default to "Unknown" if there's an issue
    return labels[predicted_index]


# Open the camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Process the frame with MediaPipe hand tracking
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the landmarks (dots) on the hand
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Preprocess the frame for model prediction
            preprocessed_frame = preprocess_frame(frame, hand_landmarks.landmark)
            
            # Make predictions
            prediction = model.predict(preprocessed_frame)
            
            # Get the label
            label = get_label(prediction)
            
            # Display the label on the frame
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Display the resulting frame
    cv2.imshow('Sign Language Recognition', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
