# ro-stories dataset

Dataset available at: https://huggingface.co/datasets/readerbench/ro-stories

# ro-auth-detect

![AA](https://github.com/readerbench/ro-auth-detect/blob/5acf9864552927864e58dca2b6b5f73f9f87b477/AA_method.png)

# content

- feature_extraction_selection.py - to extract ReaderBench Textual Complexity Indices and select top top 100 indices via Kruskal-Wallis Mean Rank
- ensemble_learning_ft.py & ensemble_learning_pp.py - combines 7 ML models with soft vote function to generate improved predictions (ft-full text; pp-paragraphs)
- hybrid_transformer_ft.py & hybrid_transformer_pp.py - hybrid transformer model, combining text inputs and linguistic features to predict the author of a given text (ft-full text; pp-paragraphs)

# usage

- the code was tested using Google Colab Pro, which is recommended for visualizations
- small adjustments to the files may be required to execute them as Python scripts
- detailed instructions and requirements are provided within each file


# paper

Cite this as: ...
(paper to be uploaded soon)
