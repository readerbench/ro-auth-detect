'''
Feature extraction - Readerbech textual complexity indices
Feature selection - Kruskal Wallis mean rank

Pre-req: make sure you have Readerbench and spacy installed and the latest numpy version

!pip install spacy
!python3 -m spacy download ro_core_news_lg
!pip install git+https://github.com/readerbench/ReaderBench.git
!git clone https://github.com/readerbench/ReaderBench.git
%cd ReaderBench
!pip install .
'''

import pandas as pd
from scipy.stats import kruskal
from rb import Document, Lang
from rb.cna.cna_graph import CnaGraph
from rb.complexity.complexity_index import compute_indices
from rb.similarity.vector_model_factory import (VectorModelType, create_vector_model)

df=pd.read_excel('ro_fulltext.xlsx')
lang = Lang.RO
results = []

for _, row in df.iterrows():
    author = row['author']
    title = row['title']
    paragraph = row['text']

    doc = Document(lang, paragraph)
    model = create_vector_model(lang, VectorModelType.TRANSFORMER, "")
    model.encode(doc)
    cna_graph = CnaGraph(docs=doc, models=[model])
    compute_indices(doc=doc, cna_graph=cna_graph)

    # Store the results for this paragraph in the results list
    result = {
        'author': author,
        'title': title,
        **{
            str(index): float(value) if value is not None else None
            for index, value in doc.indices.items()
        }
    }

    results.append(result)

indices_df = pd.DataFrame(results)
indices_df.to_csv('fulltext_rbi.csv', index=False, encoding = 'utf-8-sig')

index_columns = indices_df.columns[2:]

results = []

for index_column in index_columns:
    groups = [indices_df[index_column][indices_df['author'] == author] for author in indices_df['author'].unique()]
    if any(len(set(group)) > 1 for group in groups):
        stat, p_value = kruskal(*groups)
        results.append((index_column, stat, p_value))

# based on statistic 
results.sort(key=lambda x: x[1], reverse=True) 

# based on p-value
#results.sort(key=lambda x: x[2], reverse=True)

top_results = results[:100]
for i, (index_column, stat, p_value) in enumerate(top_results, start=1):
    print(f"{i}. Index: {index_column}, Statistic: {stat}, p-value: {p_value}")
