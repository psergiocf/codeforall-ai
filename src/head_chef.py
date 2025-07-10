import gradio as gr
from llm import query_llm
from recipes_loader import load_recipes
from recipes_chunker import chunk_recipes
from recipes_db import create_vector_db, add_documents_to_vector_db, get_relevant_chunks


vector_db = create_vector_db()


################################################################
########## Phase 1: Load information and prepare RAG ###########
################################################################
def prepare_rag():
    # Load recipes from JSON file
    recipes = load_recipes()

    # Transform recipes into chunks
    # Important Note: This step is not needed because load_recipes() uses JSONLoader, which automatically generates a document per JSON object.
    # chunks = chunk_recipes(recipes)

    # Embed the chunks and store them in a vector database
    add_documents_to_vector_db(vector_db, recipes)


################################################################
############# Phase 2: RAG Retrieval Augmentation ##############
################################################################
def predict(query, history):
    relevant_chunks = get_relevant_chunks(vector_db, query)

    return query_llm(query, relevant_chunks)


# Load the recipes and prepare RAG with the required chunks
prepare_rag()

gr.ChatInterface(predict, type="messages").launch(debug=True)
