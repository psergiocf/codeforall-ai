import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Generate embedding model
# https://python.langchain.com/docs/integrations/text_embedding/openai/
def generate_embedding_model():
    return OpenAIEmbeddings(
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        model="text-embedding-3-small"
    )

# Create a vector database using Chroma
def create_vector_db():
    embedding_model = generate_embedding_model()
    vector_db = Chroma(
        collection_name="head-chef-recipes",
        embedding_function=embedding_model,
        persist_directory="vector_db"
    )

    return vector_db

# Add documents to the vector database
def add_documents_to_vector_db(vector_db, recipes):
    # This is where the embedding happens, the vectors are stored and there are associated costs
    vector_db.add_documents(recipes)

    print(f"Vector DB: {vector_db}")

# Retrieve relevant chunks based on user query
def get_relevant_chunks(vector_db, queries, top_k=3):
    distance_threshold = 1.5
    relevant_chunks = ""

    for query in queries:
        chunks = vector_db.similarity_search_with_score(query, top_k)

        for chunk in chunks:
            if chunk[1] < distance_threshold:
                relevant_chunks = relevant_chunks + "\n" + chunk[0].page_content

    return relevant_chunks
