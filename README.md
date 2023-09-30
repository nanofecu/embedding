## Advancing Materials Synthesis Research: A Novel Approach to Chemical Reagent Identification through Word Embeddings - A Case Study on FeCu Synthesis

​      This study constructed a specialized corpus centered around ‘Fe, Cu, synthesis.’ Using this corpus, domain-specific word vector models were trained employing Natural Language Processing (NLP) technology within an unsupervised environment. Through average cosine similarity analysis, visual analysis and synonym analysis, this study, using Bimetallic FeCu as a case, revealed the potential chemical reagents involved in the synthesis of FeCu. This demonstrates the practical application value of the embedding model and verifies the effectiveness of models in the synthesis of chemical materials.



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

The paths for the input and output files required in the code need to be set according to the program's requirements.Please set the file paths as necessary before running the program.



## 3. Contact

For any questions, please contact us at mkchan@segi.edu.my.

## 4. License

Please see the <code>[LICENSE](./LICENSE)</code> file for details.
