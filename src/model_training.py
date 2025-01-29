import pandas
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.model_selection import train_test_split 

def train_model(data_file, model_dir):
    df = pandas.read_csv(data_file)
    X = df.drop('label', axis=1).values
    y = pandas.get_dummies(df['label']).values
    
    X = X.reshape(X.shape[0], X.shape[1], 1)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    #define model
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(y.shape[1], activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    #Train model
    model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
    
    #Save model weights
    model.save(f"{model_dir}/trained_model.h5")
    print(f"Model weights saved to {model_dir}/trained_model.h5")
    
    #save model architecture
    with open(f"{model_dir}/model_architecture.json", "w") as f:
        f.write(model.to_json())
    print(f"Model architecture saved to {model_dir}/model_architecture.json")

if __name__ == "__main__":
    train_model(data_file="data/annotated/annotated_data.csv", model_dir="models")