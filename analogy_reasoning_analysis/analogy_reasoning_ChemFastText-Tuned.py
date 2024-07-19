import fasttext
import numpy as np
from tqdm import tqdm

# Load a fine-tuned FastText model from a specified path
model_path = "<Your Model Path>"
model = fasttext.load_model(model_path)

def cosine_similarity(vec, all_vecs):
    """
    Calculate cosine similarity between a single vector and an array of vectors.
    Args:
        vec (np.array): A single vector.
        all_vecs (np.array): An array of vectors.
    Returns:
        np.array: Cosine similarities.
    """
    vec_norm = np.linalg.norm(vec)
    all_vecs_norm = np.linalg.norm(all_vecs, axis=1)
    
    if vec_norm == 0 or np.any(all_vecs_norm == 0):
        return np.zeros(len(all_vecs))

    # Prevent division by zero with a small epsilon
    similarities = np.dot(all_vecs, vec) / (all_vecs_norm * vec_norm + 1e-10)
    return similarities

def find_top50_words(vec_d, model):
    """
    Find the top 5 words most similar to the provided vector.
    Args:
        vec_d (np.array): The vector for which to find similar words.
        model (fasttext.FastText._FastText): Loaded FastText model.
    Returns:
        list: Top 5 words and their cosine similarity scores.
    """
    words = model.get_words()
    all_word_vectors = np.array([model.get_word_vector(word) for word in words])
    
    similarities = cosine_similarity(vec_d, all_word_vectors)
    
    # Sort the similarities and return the top 5
    top_indices = np.argsort(similarities)[-5:]
    top_words = [(words[i], similarities[i]) for i in reversed(top_indices)]
    return top_words

def analogy_top50(a, b, c, model):
    """
    Solve the analogy task "a is to b as c is to ?" by finding the word most similar to the result vector.
    Args:
        a, b, c (str): Words forming the analogy.
        model (fasttext.FastText._FastText): Loaded FastText model.
    Returns:
        list: Top 5 words fitting the analogy.
    """
    vec_a = model.get_word_vector(a)
    vec_b = model.get_word_vector(b)
    vec_c = model.get_word_vector(c)

    vec_d = vec_b - vec_a + vec_c
    return find_top50_words(vec_d, model)

# Load the analogy dataset from a specified path
analogy_tasks = []
with open("<Your Analogy Task File Path>", "r") as file:
    for line in file:
        words = line.strip().split(" :: ")
        if len(words) == 2:
            analogy_tasks.append([w.strip() for w in words[0].split(":")] + [w.strip() for w in words[1].split(":")])

# Run the analogy task and collect results
results = []
for a, b, c, d in tqdm(analogy_tasks, desc="Processing analogies"):
    predictions = analogy_top50(a, b, c, model)
    results.append(f"[{a} : {b} :: {c} : {d}],[{predictions}]")

# Save results to a specified output file
output_path = "<Your Output File Path>"
with open(output_path, "w") as output_file:
    for result in results:
        output_file.write(result + "\n")

print(f"Analogy predictions have been saved to '{output_path}'.")
