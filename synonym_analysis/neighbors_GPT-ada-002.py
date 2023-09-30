import os
import openai
import faiss
import numpy as np
import pickle
from tqdm import tqdm
import pandas as pd


# Set your OpenAI API key
openai.api_key = '<Your-API-Key>'

# Function to get an embedding for a query
def get_embedding(query, model="text-embedding-ada-002", embeddings_dict=None):
    """
    Retrieves or computes the embedding for a given query.

    """

    if query in embeddings_dict:
        return embeddings_dict[query]

    response = openai.Embedding.create(model=model, input=query)
    embedding = response["data"][0]["embedding"]

    # Cache the embedding in the embeddings_dict
    embeddings_dict[query] = embedding
    return embedding

def build_faiss_index(embeddings):
    """
    Builds a Faiss index using the provided embeddings.
    
    :param embeddings: A list of embeddings.
    :return: A Faiss index object.
    """
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)  # using L2 distance
    index.add(np.array(embeddings).astype('float32'))
    return index


# Load or initialize embeddings dict
if os.path.exists('<Your-Embeddings-Path>'):  # Example: embeddings.pkl
    with open('<Your-Embeddings-Path>', "rb") as file:  # Example: embeddings.pkl
        embeddings_dict = pickle.load(file)
else:
    embeddings_dict = {}

# Load word_list from a file
with open('<Your-Words-List-Path>', "r", encoding="utf-8") as file:  # Example: words_list.txt
    word_list = [line.strip() for line in file]

# Fetch embeddings for words not already in embeddings_dict
new_embeddings_required = any(word not in embeddings_dict for word in word_list)
if new_embeddings_required:
    request_counter = 0
    for word in tqdm(word_list, desc="Generating Embeddings"):
        if word not in embeddings_dict:
            get_embedding(word, embeddings_dict=embeddings_dict)
            request_counter += 1

            # Save after every 1000 requests
            if request_counter % 1000 == 0:
                with open('<Your-Embeddings-Path>', "wb") as file:  # Example: embeddings.pkl
                    pickle.dump(embeddings_dict, file)

    # Save remaining embeddings (if any)
    if request_counter % 1000 != 0:
        with open('<Your-Embeddings-Path>', "wb") as file:  # Example: embeddings.pkl
            pickle.dump(embeddings_dict, file)

# Build faiss index
embeddings = [embeddings_dict[word] for word in word_list]
index = build_faiss_index(embeddings)

# Get the embedding for the query word
query_word = "hydrothermal method" # Example: ['hydrothermal method','FeSO4','CuCl2','NaBH4','EDTA']
query_embedding = np.array([get_embedding(query_word, embeddings_dict=embeddings_dict)]).astype('float32')

K = 250
distances, indices = index.search(query_embedding, K)

# Create a DataFrame to hold the data
df = pd.DataFrame(columns=['Word', 'Distance'])

# Populate the DataFrame
for k in range(K):
    df.loc[k] = [word_list[indices[0][k]], distances[0][k]]

# Save the DataFrame to Excel
with pd.ExcelWriter(f'<Your-Output-Excel-Path>') as writer:  # Example: nearest_neighbors_hydrothermal_method.xlsx
    df.to_excel(writer, sheet_name='Results', index=False)
