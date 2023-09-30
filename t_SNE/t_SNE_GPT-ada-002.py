import os
import openai
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pickle
from itertools import product

# Set up the API key
openai.api_key = 'your_api_key'  # Replace with your OpenAI API Key

# Step 1: Read the content of the files
filenames = ["./chelating agents.txt", "./reducing agents.txt", "./iron salts.txt", "./copper salts.txt", "./all chemicals involved in the dataset.txt"]

chemicals = {}
for filename in filenames:
    with open(filename, 'r', encoding='utf-8') as file:
        pure_filename = os.path.basename(filename)
        chemicals[pure_filename] = [line.strip() for line in file.readlines()]

# Step 2: Obtain embeddings for all the words
embeddings = {}
for category, chems in chemicals.items():
    pickle_filename = f"{category}_embeddings.pkl"
    if os.path.exists(pickle_filename):
        with open(pickle_filename, 'rb') as pf:
            embeddings[category] = pickle.load(pf)
    else:
        embeddings[category] = []
        for chem in tqdm(chems, desc=f"Getting embeddings for {category}"):
            res = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=chem
            )
            embeddings[category].append(res["data"][0]["embedding"])
        with open(pickle_filename, 'wb') as pf:
            pickle.dump(embeddings[category], pf)

# This program reads multiple files containing chemical names, gets their embeddings using OpenAI API,
# applies t-SNE for dimensionality reduction, and finally saves the results in different plot images
# based on different combinations of t-SNE parameters.

all_embeddings = []
labels = []
for category, emb in embeddings.items():
    all_embeddings.extend(emb)
    labels.extend([category] * len(emb))

# Define parameter grid
perplexities = [5, 10, 15, 30]
learning_rates = [120, 200, 500]
early_exaggerations = [4, 12, 30]

# Create progress bar
total_combinations = len(perplexities) * len(learning_rates) * len(early_exaggerations)
progress_bar = tqdm(total=total_combinations, desc='Grid search progress')

# Perform grid search
for perp, lr, ee in product(perplexities, learning_rates, early_exaggerations):
    # Dimensionality reduction using t-SNE
    tsne = TSNE(n_components=2, perplexity=perp, learning_rate=lr, early_exaggeration=ee)
    reduced_embeddings = tsne.fit_transform(all_embeddings)

    # Save the result plots
    plt.figure(figsize=(15, 10))

    label_map = {
        "chelating agents.txt": "Chelating agent",
        "reducing agents.txt": "Reducing agent",
        "iron salts.txt": "Iron Salt",
        "copper salts.txt": "Copper salt",
        "all chemicals involved in the dataset.txt": "All chemicals"
    }

    colors = {
        "Chelating agent": 'r',
        "Reducing agent": 'g',
        "Iron Salt": 'b',
        "Copper salt": 'y',
        "All chemicals": 'lightgray'
    }
    
    sizes = {
        "Chelating agent": 50,
        "Reducing agent": 50,
        "Iron Salt": 50,
        "Copper salt": 50,
        "All chemicals": 5
    }

    for idx, label in enumerate(labels):
        english_label = label_map[label]
        plt.scatter(reduced_embeddings[idx, 0], reduced_embeddings[idx, 1], color=colors[english_label], s=sizes[english_label])

    plt.title(f"t-SNE with perplexity={perp}, learning_rate={lr}, early_exaggeration={ee}")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.savefig(f"tsne_perp_{perp}_lr_{lr}_ee_{ee}.png")
    plt.close()

    # Update progress bar
    progress_bar.update(1)

# Close the progress bar
progress_bar.close()










