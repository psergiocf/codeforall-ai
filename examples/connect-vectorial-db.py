import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("docs/bitcoin.pdf")
docs = loader.load()
chunk_size=1000
chunk_overlap = chunk_size * 0.2 # 15% - 30%
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
chunks = text_splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'), model="text-embedding-3-small")

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
)

# no add, 1 ela vai fazer embedding de todos os chunks, e depois vais os guardar - é aqui que vamos ter os custos
# Custos = x chunks (quantos tokens tem cada chunk)
vector_store.add_documents(chunks)

# A simple similarity search (no retrieval nor LLMs at this point)

# ele aqui faz outra vez o middle step de transformar a query em embedding
# Custo = 1 chunk (tokens tem este)
query = "What is bitcoin?"
docs = vector_store.similarity_search(query, k=1)

print(f"query {query} \n")
print(f"all docs {docs} \n")
print(f"top search result {docs[0].page_content} \n")

print("-" * 80)

# check distance score
# distance score is L2 distance, lower is better

# ele aqui faz outra vez o middle step de transformar a query em embedding
# Custo = 1 chunk (tokens tem este)
docs_and_scores = vector_store.similarity_search_with_score(query)
print(f"all docs {docs_and_scores} \n")
print(f"top search result {docs_and_scores[0]} \n")