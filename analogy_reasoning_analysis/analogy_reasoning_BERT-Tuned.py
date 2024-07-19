import torch
from transformers import BertTokenizer, BertForMaskedLM
from tqdm import tqdm
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# Define the paths to the fine-tuned model and tokenizer
model_path = "<Your Model Path>"

# Load tokenizer and model using the defined path
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForMaskedLM.from_pretrained(model_path)

# Load the original BERT tokenizer for 'bert-base-uncased'
original_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Function to calculate word embeddings using the model
def get_word_embedding(word, model, tokenizer):
    """
    Generate the average embedding for a given word using the BERT model.
    Args:
        word (str): The word to be embedded.
        model (PreTrainedModel): The loaded BERT model.
        tokenizer (Tokenizer): The tokenizer corresponding to the BERT model.
    Returns:
        np.array: The average embedding vector of the word.
    """
    tokenized_word = tokenizer(word, return_tensors='pt')
    with torch.no_grad():
        outputs = model.bert(**tokenized_word)
    return outputs.last_hidden_state[0, 1:-1, :].mean(dim=0).numpy()

# File for storing word vectors
word_vectors_filename = 'word_vectors.pkl'
if not os.path.exists(word_vectors_filename):
    # Extract vocabulary from the original tokenizer
    vocab_list = list(original_tokenizer.get_vocab().keys())
    # Compute and store word vectors in a dictionary
    word_vectors = {}
    for word in tqdm(vocab_list, desc="Generating embeddings"):
        word_vectors[word] = get_word_embedding(word, model, tokenizer)
    # Save word vectors to a file
    with open(word_vectors_filename, 'wb') as f:
        pickle.dump(word_vectors, f)

# Load word vectors from the previously saved file
with open(word_vectors_filename, 'rb') as f:
    loaded_word_vectors = pickle.load(f)

# Define a function for analogy reasoning
def analogy(word1, word2, word3, topn=5):
    """
    Perform analogy reasoning to find words that complete the analogy: word1 is to word2 as word3 is to ?
    Args:
        word1, word2, word3 (str): Words forming the base of the analogy.
        topn (int): Number of top results to return.
    Returns:
        list: Top 'topn' words and their similarity scores.
    """
    vec1 = loaded_word_vectors[word1]
    vec2 = loaded_word_vectors[word2]
    vec3 = loaded_word_vectors[word3]
    analogy_vector = vec2 - vec1 + vec3
    similarities = []
    for word, vector in loaded_word_vectors.items():
        sim = cosine_similarity([analogy_vector], [vector])[0][0]
        similarities.append((word, sim))
    sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    filtered_similarities = [(word, sim) for word, sim in sorted_similarities if word not in [word1, word2, word3]]
    return filtered_similarities[:topn]

# Read and process analogy tasks from a file
file_path = "<Your Input File Path>"
results_to_write = []
line_count = sum(1 for line in open(file_path, 'r'))

with open(file_path, 'r') as file:
    for line_number, line in tqdm(enumerate(file, 1), total=line_count, desc="Processing analogies"):
        parts = line.strip().split(' :: ')
        if len(parts) == 2:
            pair1 = parts[0].split(' : ')
            pair2 = parts[1].split(' : ')
            if len(pair1) == 2 and len(pair2) == 2:
                word1, word2, word3, expected = pair1[0], pair1[1], pair2[0], pair2[1]
                if all(word in loaded_word_vectors for word in [word1, word2, word3]):
                    results = analogy(word1, word2, word3, topn=50)
                    results_to_write.append(f"[{word1} : {word2} :: {word3} : {expected}],[{results}]")
                else:
                    print(f"Skipping line {line_number} due to missing words.")

# Write the results to a specified output file
results_file_path = "<Your Output File Path>"

with open(results_file_path, 'w') as file:
    for result in results_to_write:
        file.write(result + '\n')

print(f"Processed {len(results_to_write)} analogies. Results saved to '{results_file_path}'.")
