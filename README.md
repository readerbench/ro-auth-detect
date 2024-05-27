# ro-stories dataset

Dataset available at: https://huggingface.co/datasets/readerbench/ro-stories

# ro-auth-detect

![AA](https://github.com/readerbench/ro-auth-detect/blob/f555ac300ee4f4f24680e7d34392aeeccf72790e/AA-method.png)

# content

- feature_extraction_selection.py - to extract ReaderBench Textual Complexity Indices and select top top 100 indices via Kruskal-Wallis Mean Rank
- ensemble_learning_ft.py & ensemble_learning_pp.py - combines 7 ML models with soft vote function to generate improved predictions (ft-full text; pp-paragraphs)
- hybrid_transformer_ft.py & hybrid_transformer_pp.py - hybrid transformer model, combining text inputs and linguistic features to predict the author of a given text (ft-full text; pp-paragraphs)
- stats directory contains the results of McNemar's and Cochran's Q statistical tests

# usage

- the code was tested using Google Colab Pro, which is recommended for visualizations
- small adjustments to the files may be required to execute them as Python scripts
- detailed instructions and requirements are provided within each file


# paper

Cite this as: Nitu M, Dascalu M. Authorship Attribution in Less-Resourced Languages: A Hybrid Transformer Approach for Romanian. Applied Sciences. 2024; 14(7):2700. https://doi.org/10.3390/app14072700
