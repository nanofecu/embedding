import os
import fasttext
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# 1. Read the content of the files
# Example file paths are provided as comments.
file_paths = [
    "<Your_Path>/chelating agents.txt",  # e.g. "./chelating agents.txt"
    "<Your_Path>/reducing agents.txt",  # e.g. "./reducing agents.txt"
    "<Your_Path>/iron salts.txt",  # e.g. "./iron salts.txt"
    "<Your_Path>/copper salts.txt",  # e.g. "./copper salts.txt"
    "<Your_Path>/all chemicals involved in the dataset.txt"  # e.g. "./all chemicals involved in the dataset.txt"
]

chemicals = {}
for file_path in file_paths:
    with open(file_path, 'r', encoding='utf-8') as file:
        pure_filename = os.path.basename(file_path)
        chemicals[pure_filename] = [line.strip() for line in file.readlines()]

# Load all fastText models
model_dir = "<Your_Model_Directory>"  # e.g. "/models"
model_files = [f for f in os.listdir(model_dir) if f.endswith(".bin")]

for model_file in tqdm(model_files, desc="Processing models"):
    model_path = os.path.join(model_dir, model_file)
    model = fasttext.load_model(model_path)
    
    embeddings = {}
    for category, chems in chemicals.items():
        embeddings[category] = [model.get_word_vector(chem) for chem in chems]
    
    all_embeddings = []
    labels = []
    for category, emb in embeddings.items():
        all_embeddings.extend(emb)
        labels.extend([category] * len(emb))

    # Convert all embeddings to a NumPy array for easier manipulation
    all_embeddings = np.array(all_embeddings)

    # Use t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, perplexity=15, learning_rate=120, early_exaggeration=12)
    reduced_embeddings = tsne.fit_transform(all_embeddings)

    # Plot the visualization graph
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
    
    for category, emb in embeddings.items():
        for idx, single_embedding in enumerate(emb):
            english_label = label_map[category]
            plt.scatter(single_embedding[0], single_embedding[1], color=colors[english_label], s=sizes[english_label])
            plt.annotate(chemicals[category][idx][:5], 
                         (single_embedding[0], single_embedding[1]),
                         fontsize=8, alpha=0.7)

    plt.title(f"t-SNE using {model_file}")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    
    # Save the result graph
    dir = "<Your_Result_Directory>"  # e.g. "/fig"
    save_path = os.path.join(dir, f"tsne_{model_file}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

