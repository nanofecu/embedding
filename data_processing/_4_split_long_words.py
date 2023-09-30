import wordninja
from tqdm import tqdm

def process_long_words(text, min_length=20):
    """
    This function processes a string and splits the words 
    longer than the specified minimum length using wordninja.
    
    :param text: A string containing space-separated words.
    :type text: str
    :param min_length: The minimum length to consider a word for splitting.
    :type min_length: int
    :return: A string with long words processed.
    :rtype: str
    """
    words = text.split()  # Split the text into words based on space
    for i, word in enumerate(words):
        if len(word) > min_length:
            split_text = wordninja.split(word)
            words[i] = " ".join(split_text)
    return " ".join(words)

# This program reads a file, processes long words in it using wordninja
# and then writes the processed lines back to a new file.

# Open the source file to read
with open('<Your Input File Path>', 'r', encoding='utf-8') as infile:
    lines = infile.readlines()

# Apply wordninja to long words
processed_lines = [process_long_words(line.strip()) for line in tqdm(lines, desc="Processing lines")]

# Write the processed lines to a new file
with open('<Your Output File Path>', 'w', encoding='utf-8') as outfile:
    for line in processed_lines:
        outfile.write(line + '\n')
