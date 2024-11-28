import os
import pandas as pd
import sys

# Define the folder containing the .txt files
folder_path = "./CVNN_VGG"  # Replace with the actual path

# Initialize an empty list to store data
data = []

# Loop over each file in the folder
for filename in os.listdir(folder_path):
    if filename.startswith("evaluation_results_") and filename.endswith(".txt"):
        # Extract the layer setting from the filename
        layer_setting = filename.replace("evaluation_results_", "").replace(".txt", "")

        # Open and read the file
        with open(os.path.join(folder_path, filename), 'r') as file:
            lines = file.readlines()
            # Extract error and ece from lines
            error = float(lines[0].strip().split(": ")[1])
            ece = float(lines[1].strip().split(": ")[1])
            # Calculate accuracy
            accuracy = 100 - error

            # Append the data
            data.append({
                "layer_setting": layer_setting,
                "accuracy": accuracy,
                "ece": ece
            })

# Create a DataFrame from the data list
df = pd.DataFrame(data)

# Save DataFrame to CSV
output_path = "./CVNN_VGG/collected_data.csv"  # Replace with desired output path
df.to_csv(output_path, index=False)

print("Data collection and CSV creation complete. File saved at:", output_path)
