import fasttext
import time
from tqdm import tqdm


''''
The script loads a pre-trained FastText model to compute similarity scores between word pairs listed in the file.
'''


# Load the model
model = fasttext.load_model('<Your_Model_File_Path>')  # Example: 'cc.en.300.bin'

# Read the specified test set from the file
with open('<Your_Input_File_Path>', 'r', encoding='utf-8') as f:  # Example: './D1.txt'
    lines = f.readlines()

# Dictionary to store the computed similarity scores
similarity_scores = {}

# Using tqdm to wrap your loop, this will display a progress bar
for line in tqdm(lines, desc="Processing", ncols=100):
    # Extract compound names
    words = eval(line.strip().split('. ')[1])
    word1, word2 = words
    print('First word is:', word1)
    print('Second word is:', word2)

    print(model.get_nearest_neighbors(word1, k=50))

    # Use the model to compute the similarity between two words
    word1_vector = model.get_word_vector(word1)
    word2_vector = model.get_word_vector(word2)
    
    # Compute cosine similarity
    cosine_similarity = (sum(a*b for a, b in zip(word1_vector, word2_vector))) / ((sum(a*a for a in word1_vector)**0.5) * (sum(b*b for b in word2_vector)**0.5))
    similarity_scores[f"{word1} - {word2}"] = cosine_similarity

    print("similarity_scores:", similarity_scores)


# Save the results to a txt file
with open("<Your_Output_File_Path>", "w", encoding="utf-8") as outfile:  # Example: "./D1_res.txt"
    for key, value in similarity_scores.items():
        outfile.write(f"Cosine Similarity for {key}: {value:.4f}\n")
    
    # Compute average cosine similarity
    average_similarity = sum(similarity_scores.values()) / len(similarity_scores)
    outfile.write(f"Average Cosine Similarity is: {average_similarity:.4f}\n")
