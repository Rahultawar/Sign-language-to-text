import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report

def load_trained_model(model_dir):
    # Load model weights
    model = load_model(f"{model_dir}/trained_model.h5")
    
    # Load model architecture
    with open(f"{model_dir}/model_architecture.json", "r") as f:
        model_architecture = f.read()
    model = tf.keras.models.model_from_json(model_architecture)
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def test_model(model, test_data_file):
    df = pd.read_csv(test_data_file)
    X_test = df.drop('label', axis=1).values
    y_test = pd.get_dummies(df['label']).values
    
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
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
    test_data_file = "data/test/test_data.csv"  # Use the saved test data file
    model = load_trained_model(model_dir)
    test_model(model, test_data_file)