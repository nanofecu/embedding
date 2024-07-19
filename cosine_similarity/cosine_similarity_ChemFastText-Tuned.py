import fasttext
import time
from tqdm import tqdm

''''
The script loads a pre-trained FastText model to compute similarity scores between word pairs listed in the file.
'''

# Load the model
# Example: '/fine_tuned_chemistry_model_cbow.bin'
model = fasttext.load_model('<Your Model Path>')

# Read the specified test set from a file
# Example: './D1.txt'
with open('<Your Test Set File Path>', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# A dictionary to store the computed similarity values
similarity_scores = {}

# Wrap your loop with tqdm for progress bar display
for line in tqdm(lines, desc="Processing", ncols=100):
    # Extracting compound names
    words = eval(line.strip().split('. ')[1])
    word1, word2 = words
    print('First word is:', word1)
    print('Second word is:', word2)

    print(model.get_nearest_neighbors(word1, k=50))

    # Use the model to compute the similarity between two words
    word1_vector = model.get_word_vector(word1)
    word2_vector = model.get_word_vector(word2)
    
    # Compute Cosine Similarity
    cosine_similarity = (sum(a*b for a, b in zip(word1_vector, word2_vector))) / ((sum(a*a for a in word1_vector)**0.5) * (sum(b*b for b in word2_vector)**0.5))
    similarity_scores[f"{word1} - {word2}"] = cosine_similarity

    print("similarity_scores:", similarity_scores)

# Save the results to a txt file
# Example: './D4_res.txt'
with open('<Your Output File Path>', "w", encoding="utf-8") as outfile:
    for key, value in similarity_scores.items():
        outfile.write(f"Cosine Similarity for {key}: {value:.4f}\n")
    
    # Compute average cosine similarity
    average_similarity = sum(similarity_scores.values()) / len(similarity_scores)
    outfile.write(f"The average cosine similarity is: {average_similarity:.4f}\n")

