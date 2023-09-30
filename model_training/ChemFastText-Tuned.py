import os
import fasttext

# Load Pretrained Model
pretrained_model_path = '<Your Path to Pretrained Model>'  # e.g. 'cc.en.300.bin'
pretrained_model = fasttext.load_model(pretrained_model_path)

# Retrieve the vocabulary from the pretrained model
words = pretrained_model.get_words()

# Initialize the vocabulary of the new model with the word vectors from the pretrained model
with open('init_vectors.vec', 'w') as f:
    f.write(str(len(words)) + " " + str(pretrained_model.get_dimension()) + "\n")
    for w in words:
        v = pretrained_model.get_word_vector(w)
        v_str = " ".join([str(x) for x in v])
        f.write(w + " " + v_str + "\n")

num_cores = os.cpu_count()
num_threads = int(0.8 * num_cores)  # Use 80% of the core count

# Fine-Tuning Parameters
params = {
    "input": '<Your Path to Input Text>',  # e.g. 'data.txt'
    "model": 'cbow',
    "lr": 0.001,
    "dim": 300,
    "ws": 10,
    "epoch": 60,
    "minCount": 3,
    "wordNgrams": 2,
    "minn": 2,
    "maxn": 7,
    "thread": num_threads,
    "pretrainedVectors": 'init_vectors.vec'  # Initial vectors
}

# This program loads a pretrained FastText model and a text file.
# It initializes a new model with the vectors from the pretrained model,
# performs unsupervised learning fine-tuning on the given text,
# and saves the fine-tuned model.
# Additionally, the program cleans up the initial vectors file after saving the model.

# Perform fine-tuning using unsupervised learning
model = fasttext.train_unsupervised(**params)

# Save the fine-tuned model
model.save_model("fine_tuned_chemistry_model_cbow.bin")
print("Model has been fine-tuned and saved!")

# Clean up the initial vectors file
os.remove('init_vectors.vec')