import os
import fasttext
from tqdm import tqdm


def train_fasttext_model(input_path, output_dir, model_type='cbow', lr=0.05, dim=300, ws=5, epoch=5,
                         min_count=1, word_ngrams=1, minn=2, maxn=6, neg=5, thread=None,
                         loss='ns', bucket=2000000, t=1e-4):
    """
    This function trains a FastText model with the provided parameters and saves it to the specified directory.
    :param input_path: Path to the input text file used for training the model.
    :param output_dir: Directory where the trained model will be saved.
    :param model_type: Model type ('cbow' or 'skipgram').
    :param lr: Learning rate.
    :param dim: Size of word vectors.
    :param ws: Size of the context window.
    :param epoch: Number of epochs.
    :param min_count: Minimal number of word occurrences.
    :param word_ngrams: Max length of word ngram.
    :param minn: Min length of char ngram.
    :param maxn: Max length of char ngram.
    :param neg: Number of negatives sampled.
    :param thread: Number of threads.
    :param loss: Loss function.
    :param bucket: Number of buckets.
    :param t: Sampling threshold.
    :return: Path where the model is saved.
    """
    if thread is None:
        num_cores = os.cpu_count()
        thread = int(0.8 * num_cores)  # Using 80% of available cores

    model = fasttext.train_unsupervised(
        input=input_path,
        model=model_type,
        lr=lr,
        dim=dim,
        ws=ws,
        epoch=epoch,
        minCount=min_count,
        wordNgrams=word_ngrams,
        minn=minn,
        maxn=maxn,
        neg=neg,
        thread=thread,
        loss=loss,
        bucket=bucket,
        t=t
    )

    model_name = f"{model_type}_lr{lr}_dim{dim}_ws{ws}_epoch{epoch}_mc{min_count}_wn{word_ngrams}_minn{minn}_maxn{maxn}.bin"
    model_path = os.path.join(output_dir, model_name)
    model.save_model(model_path)

    return model_path


# Grid search parameters
model_types = ['cbow', 'skipgram']
lrs = [0.01, 0.001, 0.0001]
epochs = [30, 60]
wss = [5, 10]
min_counts = [1, 3]
word_ngrams = [1, 2]
minns = [2]
maxns = [7]
dims = [300]

input_path = '<Your Input File Path>'
output_dir = '<Your Output Directory Path>'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Calculate total number of models for setting the progress bar's total length
total_models = len(model_types) * len(lrs) * len(dims) * len(wss) * len(epochs) * len(min_counts) * len(word_ngrams) * len(minns) * len(maxns)

# Setting up a progress bar with tqdm
with tqdm(total=total_models, desc="Training Progress") as pbar:
    for model_type in model_types:
        for lr in lrs:
            for dim in dims:
                for ws in wss:
                    for epoch in epochs:
                        for min_count in min_counts:
                            for wn in word_ngrams:
                                for minn in minns:
                                    for maxn in maxns:
                                        model_path = train_fasttext_model(
                                            input_path=input_path,
                                            output_dir=output_dir,
                                            model_type=model_type,
                                            lr=lr,
                                            dim=dim,
                                            ws=ws,
                                            epoch=epoch,
                                            min_count=min_count,
                                            word_ngrams=wn,
                                            minn=minn,
                                            maxn=maxn
                                        )
                                        pbar.update(1)  # Update progress bar
