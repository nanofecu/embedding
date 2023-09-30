import os
import fasttext
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE



# 1. Load the content of the files.
# Replace example_filenames with the paths of your files.
filenames = ["./chelating agents.txt", "./reducing agents.txt", "./iron salts.txt", "./copper salts.txt", "./all chemicals involved in the dataset.txt"]

chemicals = {}
for filename in filenames:
    with open(filename, 'r', encoding='utf-8') as file:
        pure_filename = os.path.basename(filename)
        chemicals[pure_filename] = [line.strip() for line in file.readlines()]

# Load all fastText models
# Replace with your model directory path.
model_dir = "<Your Model Directory Path>"  # e.g. "/models"

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

    # Convert all embeddings to a NumPy array for easier manipulation.
    all_embeddings = np.array(all_embeddings)

    # Perform dimensionality reduction using t-SNE
    tsne = TSNE(n_components=2, perplexity=15, learning_rate=120, early_exaggeration=12)
    reduced_embeddings = tsne.fit_transform(all_embeddings)

    # Plotting the visualization
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

    plt.title(f"t-SNE using {model_file}")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    
    # Save the resultant plot
    # Replace with your directory path.
    dir = "<Your Results Directory Path>"  # Example: save_dir = '/fig'
    save_path = os.path.join(dir, f"tsne_{model_file}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
