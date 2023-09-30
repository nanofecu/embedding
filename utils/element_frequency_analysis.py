import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

'''
The overall functionality of this code is to read the content of a file, 
analyze the length and frequency of the words,
and perform frequency analysis on the elements in the given element list. Finally, 
the analysis results are saved to files and images.
'''


# Step 1: Read the content of the file
# Replace 'your_input_file_path' with the path of your input file
with open('your_input_file_path', "r", encoding="utf-8") as f:
    lines = f.readlines()
    # Convert each line to a list of words and calculate the length
    lengths = [len(line.strip().split()) for line in lines if line.strip()]

# Step 2: Calculate the frequency of each length using Counter
length_frequencies = Counter(lengths)

# Step 3: Sort by sentence length
sorted_length_freq = sorted(length_frequencies.items(), key=lambda x: x[0])

# Step 4: Save length and frequency to a txt file
# Replace 'your_output_file_path' with the path where you want to save the output file
with open('your_output_file_path', "w", encoding="utf-8") as out_file:
    for length, freq in sorted_length_freq:
        out_file.write(f"{length},{freq}\n")

print("Length frequencies saved to your_output_file_path")

# Step 5: Read the content of the file again and split the content into words
# Then, calculate the frequency of each word and save the result to a file
with open('your_input_file_path', 'r', encoding='utf-8') as file:
    content = file.read()

words = content.split()
word_freq = Counter(words)

with open('word_frequency_file_path', 'w', encoding='utf-8') as file:
    for word, freq in word_freq.items():
        file.write(f"{word},{freq}\n")

# Step 6: Perform frequency analysis on a given list of elements and save the result
# The result is also plotted and saved as an image

ELEMENTS = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K",
            "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
            "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",
            "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
            "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr",
            "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf",
            "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og", "Uue"]

word_freq = {}

# Read the word frequency from the file
with open("word_frequency_file_path", "r", encoding="utf-8") as file:
    lines = file.readlines()
    for line in lines:
        try:
            word, freq = line.split(',')
            word_freq[word] = int(freq.strip())
        except ValueError:
            print(f"Problematic line: {line.strip()}")

# Select elements that appear in the word frequency file and record their frequencies
selected_elements_freq = {element: word_freq[element] for element in ELEMENTS if element in word_freq}

# Plot the frequency analysis of elements using matplotlib
plt.figure(figsize=(20, 8))
plt.bar(selected_elements_freq.keys(), selected_elements_freq.values(), color='skyblue')
plt.xlabel('Element')
plt.ylabel('Frequency')
plt.title('Element Frequency Analysis')
plt.xticks(rotation=90)
plt.tight_layout()

# Save the selected elements and their frequencies to a new file
with open("selected_elements_frequency_file_path", "w", encoding="utf-8") as file:
    for element, freq in selected_elements_freq.items():
        file.write(f"{element},{freq}\n")

# Save the frequency analysis graph as an image
plt.savefig("element_frequency_analysis_image_path")

