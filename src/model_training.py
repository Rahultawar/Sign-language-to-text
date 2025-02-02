import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.model_selection import train_test_split

def train_model(data_file, model_dir):
    # Load the dataset (using train data for now)
    df = pd.read_csv(data_file)
    X = df.drop('label', axis=1).values
    y = pd.get_dummies(df['label']).values
    
    # Reshaping X for LSTM
    X = X.reshape(X.shape[0], X.shape[1], 1)  # 3D shape (samples, features, 1)
    
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define model architecture (LSTM used for sequence data or time series)
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(y.shape[1], activation='softmax')
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Train model without data augmentation
    model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_val, y_val))
    
    # Save model weights
    model.save(f"{model_dir}/trained_model.h5")
    print(f"Model weights saved to {model_dir}/trained_model.h5")
    
    # Save model architecture
    with open(f"{model_dir}/model_architecture.json", "w") as f:
        f.write(model.to_json())
    print(f"Model architecture saved to {model_dir}/model_architecture.json")

if __name__ == "__main__":
    train_model(data_file="data/splits/train_data.csv", model_dir="models")
