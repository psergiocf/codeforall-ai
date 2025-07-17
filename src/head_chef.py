import gradio as gr
from llm import generate_multiple_queries, query_llm, split_user_query_into_queries
from recipes_loader import load_recipes
from recipes_chunker import update_recipes_data
from recipes_db import create_vector_db, add_documents_to_vector_db, get_relevant_chunks

vector_db = create_vector_db()

################################################################
########## Phase 1: Load information and prepare RAG ###########
################################################################
def ingestion_stage():
    # Only add documents if the vector database is empty
    if vector_db._collection.count() > 0:
        print("⚠️  Vector DB already has recipes, skipping embedding.")
        return

    # Load recipes from JSON file
    recipes = load_recipes()

    # Since the loaded recipes are automatically returned as documents which are our chunks.
    # Due to this, we don't need to create chunks manually.
    # Retrieve new recipe documents with updated metadata
    updated_recipes = update_recipes_data(recipes)

    # Embed the chunks and store them in a vector database
    add_documents_to_vector_db(vector_db, updated_recipes)


################################################################
############# Phase 2: RAG Retrieval Augmentation ##############
################################################################
def inference_stage(query, history):
    splitted_user_query = split_user_query_into_queries(query)
    generated_queries = generate_multiple_queries(splitted_user_query)

    print(f"Splitted queries: {splitted_user_query}\n")
    print(f"Generated queries: {generated_queries}\n")

    relevant_chunks = get_relevant_chunks(vector_db, query, generated_queries)

    return query_llm(query, relevant_chunks)


# Load the recipes and prepare RAG with the required chunks
ingestion_stage()

gr.ChatInterface(inference_stage, type="messages").launch(debug=True)
