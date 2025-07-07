import json
import os
# import gradio as gr
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# def predict(message, history):
# Phase 1: Load information and prepare RAG

####### TODO: Move this logic to a file/DB loader module
# Load the JSON file with recipes
loader = JSONLoader(
    file_path="docs/shelf/recipes.json",
    jq_schema=".[]",
    text_content=False,
)

# Load the documents
documents = loader.load()

# Each object becomes a separate document
for document in documents:
    page_content = json.loads(document.page_content)
    doc_id = page_content.get('id')
    # print(f"Document: {document}")
    # print("---")
    # print(f"ID: {doc_id}")
    # print("---------------------------")
####### END OF TODO






####### TODO: Move this logic to a transformation module
# Document transformation - https://python.langchain.com/docs/how_to/#text-splitters
chunk_size = 500 # 500 - 1000 andam de 100 em 100
chunk_overlap = chunk_size * 0.2 #15% a 30% do chunk size - não existe em texto estruturado
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = chunk_size, chunk_overlap = chunk_overlap
)

chunks = text_splitter.split_documents(documents)

for chunk in chunks:
    print(f"Chunks: {chunk}")
    print("---------------------------")
####### END OF TODO





####### TODO: Move this logic to a embedding module
# Step 1.3 - Embed - https://python.langchain.com/docs/integrations/text_embedding/openai/ - custos
# Custos dependentes de quantos tokens temos nos nossos chunks
# NÃO SE ESQUEÇAM QUE ESTA FASE EXISTE, E TEM CUSTOS, SIMPLESMENTE ESTÁ INTEGRADA NA PRÓXIMA
embedding_model = OpenAIEmbeddings(
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    model = "text-embedding-3-small"
)

print("1 ---------------------------")

# # transformar chunks -> quantos tokens tem - 1M tokens custa 0.02$ - https://github.com/openai/tiktoken

# for chunk in chunks:
#     embedded_chunk = embedding_model.embed_documents([chunk.page_content])
#     print(f"Embedded Chunk: {embedded_chunk}")
#     print("---------------------------")
# ####### END OF TODO






####### TODO: Move this logic to a embedding module
# Step 1.4 - Guardar na base de dados vectorial - https://python.langchain.com/docs/integrations/vectorstores/
vector_db = Chroma(
    collection_name="head-chef-recipes",
    embedding_function=embedding_model
)

print("2 ---------------------------")

# Não esquecer que é nesta fase que acontece o embedding e depois os vetores são guardados, CUSTOS
vector_db.add_documents(chunks)

print("3 ---------------------------")

print(f"Vector DB: {vector_db}")
print("---------------------------")
# ####### END OF TODO



# return "Welcome to the Head Chef AI!"

# gr.ChatInterface(predict, type="messages").launch(debug=True)





#################### KaggleHub Example ##########################
#import kagglehub

# Download latest version
#path = kagglehub.dataset_download("crispen5gar/recipes3k")

#print(f"Dataset downloaded to: {path}")
##################################################################