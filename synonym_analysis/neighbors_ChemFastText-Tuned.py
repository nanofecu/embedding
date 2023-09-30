import fasttext
import pandas as pd


model_path = "<Your Model Path>"  # Example: "/fine_tuned_chemistry_model_cbow.bin"
model = fasttext.load_model(model_path)

# This program loads a pre-trained FastText model and uses it to find the nearest neighbors
# for a list of specified words. The neighbors and their similarity are then saved to Excel files.

word_lists = ['hydrothermal method', 'FeSO4', 'CuCl2', 'NaBH4', 'EDTA']

for index_, word_i in enumerate(word_lists):
    try:
        # Get the nearest neighbors for the word
        neighbors = model.get_nearest_neighbors(word_i, k=250)
        print("neighbors:", neighbors)
        
        # Create an empty list to store neighbor data
        data_list = []
        
        # Extract neighbor data and add it to the list
        for neighbor in neighbors:
            similarity, word = neighbor
            data_list.append({"Similarity": similarity, "Word": word})
        
        # Create a DataFrame from the list
        df = pd.DataFrame(data_list)
        
        # Save the DataFrame to an Excel file
        df.to_excel(f"neighbors_{word_i}_{index_}.xlsx", index=False)

    except Exception as e:
        print(f"Error processing {word_i}: {e}")

