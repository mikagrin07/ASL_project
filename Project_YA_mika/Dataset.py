import numpy as np
from PIL import Image, ImageEnhance
import random
import os
import pandas as pd
import pillow_heif



# Path to the main folder containing the 26 subfolders
main_folder_path = r"C:\Users\User\Downloads\Dataset_signLetters\DataWithMan"
# Path to save the CSV file
csv_file_path = r'C:\Users\User\Downloads\Dataset_signLetters\ManDataHome1.csv'

# List to store the image data and labels
data = []

# Register the HEIF plugin with Pillow
pillow_heif.register_heif_opener()

# Iterate through each subfolder in the main folder
for subfolder in os.listdir(main_folder_path):
    if subfolder in ["Space", "Nothing"]:
        continue  # Skip this subfolder
    subfolder_path = os.path.join(main_folder_path, subfolder)
    if os.path.isdir(subfolder_path):
        # Iterate through each inner subfolder
        for inner_subfolder in os.listdir(subfolder_path):
            inner_subfolder_path = os.path.join(subfolder_path, inner_subfolder)
            if os.path.isdir(inner_subfolder_path):
                # Process each image in the inner subfolder
                for image_file in os.listdir(inner_subfolder_path):
                    image_path = os.path.join(inner_subfolder_path, image_file)
                    if os.path.isfile(image_path):
                        try:
                            # Load the image, HEIC files are automatically supported by Pillow with pillow_heif
                            image = Image.open(image_path).convert('L')
                            # Resize the image to 28x28
                            image = image.resize((28, 28), Image.Resampling.LANCZOS)
                            image_np = np.array(image).flatten()  # Flatten the image to 1D array
                            data.append([subfolder] + image_np.tolist())  # Add label and flattened image data
                        except Exception as e:
                            print(f"Could not process image {image_path}: {e}")

# Convert the data to a DataFrame
columns = ['label'] + [f'pixel_{i}' for i in range(28 * 28)]
df = pd.DataFrame(data, columns=columns)

# Save the DataFrame to a CSV file
df.to_csv(csv_file_path, index=False)

print(f"Data saved to {csv_file_path}")
    
'''
# Function to convert a letter to its corresponding number
def letter_to_number(letter):
    return ord(letter) - ord('A')'''

# Path to the input CSV file
#input_csv_path = r'C:\Users\User\Downloads\Dataset_signLetters\TrainDataHome1.csv'
# Path to the output CSV file
#output_csv_path = r'C:\Users\User\Downloads\Dataset_signLetters\ReadyTrainDataHome1.csv'
'''
# Read the CSV file into a DataFrame
df = pd.read_csv(input_csv_path)

# Convert the letters in the first column to numbers
df.iloc[:, 0] = df.iloc[:, 0].apply(lambda x: letter_to_number(x) if isinstance(x, str) and x.isupper() and len(x) == 1 else x)

# Save the modified DataFrame to a new CSV file
df.to_csv(output_csv_path, index=False)

print(f"Modified data saved to {output_csv_path}")
    # Function to convert a letter to its corresponding number
   # def letter_to_number(letter):
      #  return ord(letter) - ord('A')
'''
    # Paths to the input and output CSV files
    #input_csv_path = r'C:\Users\User\Downloads\Dataset_signLetters\datasetSignLanguageSubset.csv'
    #output_csv_path = r'C:\Users\User\Downloads\Dataset_signLetters\datasetSignLanguageSubsetAFT.csv'
    #first_column_csv_path = r'C:\Users\User\Downloads\Dataset_signLetters\YTrain.csv'
'''
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_csv_path)

    # Convert the letters in the first column to numbers
    df.iloc[:, 0] = df.iloc[:, 0].apply(lambda x: letter_to_number(x) if isinstance(x, str) and x.isupper() and len(x) == 1 else x)

    # Save the modified DataFrame to a new CSV file
    df.to_csv(output_csv_path, index=False)

    # Extract the first column
    first_column_df = df.iloc[:, [0]]

    # Save the first column to a new CSV file
    first_column_df.to_csv(first_column_csv_path, index=False)

    print(f"Modified data saved to {output_csv_path}")
    print(f"First column saved to {first_column_csv_path}")'''
