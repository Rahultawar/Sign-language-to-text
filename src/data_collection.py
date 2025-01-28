import cv2 #video capture and image processing
import mediapipe as mp #For hand tracking
import os #For file handling and dir. operation

def collect_data(output_dir):
    #Initialize mediapipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    cap = cv2.VideoCapture(0)
    
    if not os.path.exists('data'):
        os.makedirs('data')
        
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read() #Read frame from webcam
        if not ret:
            break
        
        # Convert the frame to RGB as MediaPipe requires RGB images
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark: #Extracting landmarks
                    landmarks.append([lm.x, lm.y, lm.z])
                    
                # Save landmarks to file
                with open(os.path.join(output_dir, f'frame_{frame_count}.csv'), 'w') as f:
                    for lm in landmarks:
                        f.write(','.join(map(str, lm)) + '\n') # Write the landmarks to a CSV file
                frame_count += 1
         
        cv2.imshow('Hand Tracking', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break     
        
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    collect_data(output_dir='data/raw')  
                