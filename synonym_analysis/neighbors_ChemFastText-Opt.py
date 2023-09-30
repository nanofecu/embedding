import fasttext
import pandas as pd



model_path = "<Your Model Path>"  # Example: /cbow_lr0.001_dim300_ws10_epoch60_mc3_wn2_minn2_maxn7.bin"

# Load the model
model = fasttext.load_model(model_path)

# List of words to find neighbors for
word_lists = ['hydrothermal method', 'FeSO4', 'CuCl2', 'NaBH4', 'EDTA']

# This program loads a specified FastText model, and for each word in the 'word_lists',
# it finds the nearest neighbors in the model's vector space, extracts the neighbors' data,
# and saves them as an Excel file.

for index_, word_i in enumerate(word_lists):
    try:
        # Get the nearest neighbors of the word
        neighbors = model.get_nearest_neighbors(word_i, k=250)
        print("neighbors:", neighbors)

        # Create an empty list to store neighbor data
        data_list = []

        # Extract neighbor data and add to the list
        for neighbor in neighbors:
            similarity, word = neighbor
            data_list.append({"Similarity": similarity, "Word": word})

        # Create a DataFrame from the list
        df = pd.DataFrame(data_list)
        df.to_excel(f"neighbors_{word_i}_{index_}.xlsx", index=False)

    except Exception as e:
        # Print error message if there is any problem processing the word
        print(f"Error processing {word_i}: {e}")

