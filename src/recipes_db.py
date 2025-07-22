import os
# import numpy
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_cohere import CohereRerank
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load the Cohere API key from environment variables which will be used for the CohereRerank class to function
os.environ["COHERE_API_KEY"] = os.getenv('COHERE_API_KEY')

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

    print(f"✅ Vector DB created.")

# Retrieve relevant chunks based on user query
def get_relevant_chunks(vector_db, original_user_query, generated_queries, top_k=3):
    distance_threshold = 1.5
    relevant_chunks = []

    for query in generated_queries:
        chunks = vector_db.similarity_search_with_score(query, top_k)

        for chunk in chunks:
            if chunk[1] < distance_threshold:
                relevant_chunks.append(chunk)
    
    return rerank_relevant_chunks(relevant_chunks, original_user_query)

def rerank_relevant_chunks(original_relevant_documents, original_user_query):
    reranked_relevant_chunks = []
    original_relevant_chunks = []

    for document in original_relevant_documents:
        original_relevant_chunks.append(document[0].page_content)

    compressor = CohereRerank(model="rerank-v3.5")
    compressed_relevant_chunks = compressor.rerank(original_relevant_chunks, original_user_query, top_n=10)

    for compressed_chunk in compressed_relevant_chunks:
        chunk = original_relevant_chunks[compressed_chunk["index"]]
        reranked_relevant_chunks.append(chunk)

    return reranked_relevant_chunks
