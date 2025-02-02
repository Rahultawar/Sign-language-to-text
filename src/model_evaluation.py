import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model, X_val, y_val, labels):
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_val, axis=1)
    
    print("Classification Report:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=labels))
    
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

if __name__ == "__main__":
    # Load the trained model
    model = load_model('models/trained_model.h5')

    # Load the validation data
    data_file = "data/splits/val_data.csv"
    df = pd.read_csv(data_file)
    X = df.drop('label', axis=1).values
    y = pd.get_dummies(df['label']).values

    # Reshape the data to match the input shape of the model
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Split the data into training and validation sets
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the labels
    labels = ['hello', 'my', 'name', 'is', 'rahul', 'other'] 

    # Evaluate the model
    evaluate_model(model, X_val, y_val, labels)