# This snippet was being used on Colab to format the output of the text splitting examples.
#
# #@title Setup better response formatting (adds line wrap)
# from IPython.display import HTML, display

# def set_css():
#   display(HTML('''
#   <style>
#     pre {
#         white-space: pre-wrap;
#     }
#   </style>
#   '''))
# get_ipython().events.register('pre_run_cell', set_css)

import os
from langchain_openai import OpenAIEmbeddings

embeddings_model = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'), model="text-embedding-3-small")

embeddings = embeddings_model.embed_documents(
    [
        "Olá",
        "Adeus",
        "Portugal",
        "Lisboa"
    ]
)
print(len(embeddings), len(embeddings[0]))

print(embeddings[:2])