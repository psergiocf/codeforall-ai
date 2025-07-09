import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Generate embedding model
# https://python.langchain.com/docs/integrations/text_embedding/openai/
# Custos dependentes de quantos tokens temos nos nossos chunks
# NÃO SE ESQUEÇAM QUE ESTA FASE EXISTE, E TEM CUSTOS, SIMPLESMENTE ESTÁ INTEGRADA NA PRÓXIMA
def generate_embedding_model():
    return OpenAIEmbeddings(
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        model="text-embedding-3-small"
    )

def generate_vector_db(embedding_model):
    return Chroma(
        collection_name="head-chef-recipes",
        embedding_function=embedding_model,
        persist_directory="vector_db"
    )

def create_vector_db(chunks):
    embedding_model = generate_embedding_model()
    vector_db = generate_vector_db(embedding_model)

    # Don't forget that this is where the embedding happens, the vectors are stored and there are associated costs
    vector_db.add_documents(chunks)

    print(f"Vector DB: {vector_db}")

    return vector_db
