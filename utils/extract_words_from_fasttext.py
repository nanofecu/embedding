import os
import fasttext

# Load the pre-trained FastText model
# Make sure to replace <Your Model Path> with the actual path to your FastText model file.
model_path = "<Your Model Path>" 
model = fasttext.load_model(model_path)


# This program loads a pre-trained FastText model and retrieves all the words in the model.
# It then writes these words to a specified output text file.

# Get all the words from the model
words = model.get_words()

# Write the words to an output txt file
# Make sure to replace <Your Output File Path> with the actual path where you want to save your words list file.
output_file_path = "<Your Output File Path>"
with open(output_file_path, 'w', encoding='utf-8') as f:
    for word in words:
        f.write(word + '\n')

print(f"Words have been saved to {output_file_path}!")


