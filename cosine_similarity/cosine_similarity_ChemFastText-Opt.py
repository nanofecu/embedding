import time
import fasttext
import os
from tqdm import tqdm


def calculate_similarity(word1, word2, model):
    """
    This function calculates the cosine similarity between two words.
    """
    word1_vector = model.get_word_vector(word1)
    word2_vector = model.get_word_vector(word2)
    
    # Calculate cosine similarity
    cosine_similarity = (sum(a*b for a, b in zip(word1_vector, word2_vector))) / (
            (sum(a*a for a in word1_vector)**0.5) * (sum(b*b for b in word2_vector)**0.5))
    return cosine_similarity

# The overall program calculates the average cosine similarity between word pairs, 
# listed in a specified test set, using multiple FastText models.

# Example Path: '/D1.txt'
test_set_path = '<Your Test Set Path>'
with open(test_set_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Example Path: "/models"
model_dir = "<Your Model Directory Path>"
model_files = [f for f in os.listdir(model_dir) if f.endswith('.bin')]

# Results will be saved to a txt file.
# Example Path: "./D4_res.txt"
output_file_path = "<Your Output File Path>"
with open(output_file_path, "w", encoding="utf-8") as outfile:
    
    # Iterate over each model file
    for model_file in tqdm(model_files, desc="Processing Models", ncols=100):
        model_path = os.path.join(model_dir, model_file)
        model = fasttext.load_model(model_path)

        similarity_scores = {}

        for line in lines:
            num = eval(line.strip().split('. ')[0])
            words = eval(line.strip().split('. ')[1])
            word1, word2 = words
            
            similarity_scores[f"{word1} - {word2}"] = calculate_similarity(word1, word2, model)
        
        # Calculate average cosine similarity
        average_similarity = sum(similarity_scores.values()) / len(similarity_scores)
        print(f"{model_file}，Average Cosine Similarity: {average_similarity:.4f}")
        time.sleep(1)

        outfile.write(f"{model_file}，Average Cosine Similarity: {average_similarity:.4f}\n")

