import os
import openai
from tqdm import tqdm
import time

# Set API key
openai.api_key = 'your_api_key'  # Replace with your OpenAI API key

# Read the specified test set from the file
# Example path: './D4.txt'
input_filepath = '<Your Input File Path>'
with open(input_filepath, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Dictionary to store the computed similarity values
similarity_scores = {}

def get_embedding(text):
    """
    This function sends a request to OpenAI API and gets the embedding vector 
    for the input text using the specified model.
    
    :param text: Input text
    :type text: str
    :return: Embedding vector for the input text
    :rtype: list
    """
    response = openai.Embedding.create(model="text-embedding-ada-002", input=text)
    return response["data"][0]["embedding"]

# Wrap your loop with tqdm to display a progress bar
for line in tqdm(lines, desc="Processing", ncols=100):
    # Extract compound names
    words = eval(line.strip().split('. ')[1])
    word1, word2 = words
    print('First word is:', word1)
    print('Second word is:', word2)

    # Obtain vectors for each word using the model
    word1_vector = get_embedding(word1)
    word2_vector = get_embedding(word2)

    # Compute Cosine Similarity
    cosine_similarity = (sum(a*b for a, b in zip(word1_vector, word2_vector))) / ((sum(a*a for a in word1_vector)**0.5) * (sum(b*b for b in word2_vector)**0.5))
    similarity_scores[f"{word1} - {word2}"] = cosine_similarity
    print("similarity_scores:", similarity_scores)
    time.sleep(0.1)

# Save the results to a txt file
# Example path: "./D1_res.txt"
output_filepath = '<Your Output File Path>'
with open(output_filepath, "w", encoding="utf-8") as outfile:
    for key, value in similarity_scores.items():
        outfile.write(f"Cosine Similarity for {key}: {value:.4f}\n")

    # Compute average Cosine Similarity
    average_similarity = sum(similarity_scores.values()) / len(similarity_scores)
    outfile.write(f"Average Cosine Similarity is: {average_similarity:.4f}\n")

