import pandas as pd
from sklearn.model_selection import train_test_split
import os

df = pd.read_csv("data/annotated/annotated_data.csv")

# Split into features (X) and labels (y)
X = df.drop('label', axis=1).values
y = pd.get_dummies(df['label']).values  # One-hot encode the labels

# Split into training, validation, and test sets (80% training, 10% validation, 10% test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Create directories for saving the data if they don't exist
os.makedirs("data/splits", exist_ok=True)
os.makedirs("data/test", exist_ok=True)

# Save the splits as CSV files
train_df = pd.DataFrame(X_train)
train_df['label'] = pd.DataFrame(y_train).idxmax(axis=1)  
train_df.to_csv("data/splits/train_data.csv", index=False)

val_df = pd.DataFrame(X_val)
val_df['label'] = pd.DataFrame(y_val).idxmax(axis=1)  
val_df.to_csv("data/splits/val_data.csv", index=False)

test_df = pd.DataFrame(X_test)
test_df['label'] = pd.DataFrame(y_test).idxmax(axis=1)  
test_df.to_csv("data/test/test_data.csv", index=False)