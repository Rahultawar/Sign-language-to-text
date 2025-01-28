import os
import pandas as pd  # For data manipulation and analysis

def annotate_data(input_dir, output_file, label):
    data = []
    labels = []
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            filepath = os.path.join(input_dir, filename)
            df = pd.read_csv(filepath, header=None)  # Read the CSV file into a dataframe
            data.append(df.values.flatten())  # Append the flattened dataframe to the data list
            labels.append(label)
            
    annotated_data = pd.DataFrame(data)  # Create a dataframe from the data list
    annotated_data['label'] = labels  # Add a column for the labels
    
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
        
    annotated_data.to_csv(output_file, index=False)  # Save the DataFrame to a CSV file without the index
    print(f"Annotated data saved to {output_file}")

if __name__ == '__main__':
    annotate_data(input_dir='data/raw', output_file='data/annotated/annotated_data.csv', label='hello')