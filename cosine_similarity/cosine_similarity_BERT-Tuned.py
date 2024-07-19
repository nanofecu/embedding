import torch
from transformers import BertTokenizer, BertForMaskedLM
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm  # Import tqdm for the progress bar


# Load the fine-tuned model and tokenizer
# Example: '/fine_tuned_bert'
model_path = '<Your Model Path>'
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForMaskedLM.from_pretrained(model_path)

# Function to get the average of embeddings for a word
def get_word_embedding(word):
    tokenized_word = tokenizer(word, return_tensors='pt')
    with torch.no_grad():
        outputs = model.bert(**tokenized_word)
    return outputs.last_hidden_state[0, 1:-1, :].mean(dim=0).numpy()  # Ignore CLS and SEP tokens

# Compute cosine similarity between two embeddings
def compute_similarity(word1, word2):
    embed1 = get_word_embedding(word1)
    embed2 = get_word_embedding(word2)
    return cosine_similarity([embed1], [embed2])[0][0]

# Process word pairs from file and calculate similarities
def process_word_pairs(file_path, output_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    similarity_scores = {}
    for line in tqdm(lines, desc="Processing word pairs"):  # Adding progress bar
        word1, word2 = line.strip().split(', ')
        sim_score = compute_similarity(word1, word2)
        similarity_scores[f"{word1} - {word2}"] = sim_score

    with open(output_path, "w", encoding="utf-8") as outfile:
        for key, value in similarity_scores.items():
            outfile.write(f"Cosine Similarity for {key}: {value:.4f}\n")
        
        average_similarity = sum(similarity_scores.values()) / len(similarity_scores)
        outfile.write(f"Average Cosine Similarity: {average_similarity:.4f}\n")

# Example usage
# '<Your Output File Path>' Example: './D1.txt'
# '<Your Output File Path>' Example: './D4_res.txt'
process_word_pairs('<Your Test Set File Path>', '<Your Output File Path>')



