## Enhancing Chemical Synthesis Research with NLP: Word Embeddings for Chemical Reagent Identification - A Case Study on nano-FeCu

​            Nanoparticles synthesis is complicated due to varied synthesis conditions and requires precise chemical reagent identification. This paper establishes a specialized "Fe, Cu, synthesis" corpus, utilizing natural language processing (NLP) to train a domain-specific word embedding model. Evaluation includes average cosine similarity, visual analysis, and synonym analysis. Findings highlight the strong correlation between learning rate and cosine similarity. t-SNE visualisation shows improved focus 
compared to models without the chemical corpus. Synonym analysis reveals the model's effectiveness in identifying potential chemical reagents for nano-FeCu particle synthesis. The study proposes a versatile interdisciplinary framework for rapid chemical reagent identification, expediting nanomaterial research and development. In conclusion, this innovative, widely applicable research method offers a data-driven pathway in chemical material synthesis.



## 1. Directories

Following is a short description of each directory under the root folder.

* <code>[data_processing](./data_processing)</code>: Functions used in data processing.
* <code>[model_training](./model_training)</code>: This function trains different models with the provided parameters and saves it to the specified directory.
* <code>[cosine_similarity](./cosine_similarity)</code>: This function loads models to compute similarity scores between word pairs listed in the file.
* <code>[t_SNE](./t_SNE)</code>: This function reads chemical data from specified files and represents each chemical with its word embedding using different models. Then reduces the dimensionality of these embeddings using t-SNE and visualizes the results in different plots.
* <code>[synonym_analysis](./synonym_analysis)</code>: This function loads a specified model, and for each word in the 'word_lists', it finds the nearest neighbors in the model's vector space, extracts the neighbors' data, and saves them as an Excel file.
* <code>[utils](./utils)</code>: Other utility files used in the project go here.

## 2. Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

![图片描述](https://github.com/nanofecu/image/blob/main/Workflow.jpg)

The paths for the input and output files required in the code need to be set according to the program's requirements. Please set the file paths as necessary before running the program.



## 3. Contact

For any questions, please contact us at mkchan@segi.edu.my.

## 4. License

Please see the <code>[LICENSE](./LICENSE)</code> file for details.
