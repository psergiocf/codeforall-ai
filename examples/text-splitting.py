# This snippet was being used on Colab to format the output of the text splitting examples.

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

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("docs/bitcoin.pdf")
docs = loader.load()

# Different chunk sizes (as well as different chunk overlaps) generate different documents
for chunk_size in [250, 500, 750, 1000]: # 500 - 1000
# for chunk_size in [10, 20, 50]:
  chunk_overlap = chunk_size * 0.2 # 15% - 30%
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  # docs = [Document(page_content=x) for x in text_splitter.split_text(text)]
  chunks = text_splitter.split_documents(docs)
  print("-" * 80)
  print(len(chunks))
  print(f"Chunk size {chunk_size} | Chunk overlap {chunk_overlap}")
  print(f"Documents: {chunks}")