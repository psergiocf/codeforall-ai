import gradio as gr
from recipes_loader import load_recipes
from recipes_chunker import chunk_recipes
from recipes_db import create_vector_db, add_documents_to_vector_db, get_relevant_chunks
from langchain_openai import ChatOpenAI         # LLM

vector_db = create_vector_db()


prompt_system = f"""
  Instructions:
  Answer the provided user query
  Use the provided context, chat history, or both to deduce the answer.
  The chat history is ordered from the first to the latest interaction, the user is a message from the user, the assistant is a message from the LLM.
  If you don't know the answer, say 'I don't know'
"""

messages = [
    ("system", prompt_system)
]


################################################################
########## Phase 1: Load information and prepare RAG ###########
################################################################
def load_information():
    # Load recipes from JSON file
    recipes = load_recipes()

    # Transform recipes into chunks
    chunks = chunk_recipes(recipes)

    # Embed the chunks and store them in a vector database
    add_documents_to_vector_db(vector_db, chunks)


################################################################
############# Phase 2: RAG Retrieval Augmentation ##############
################################################################
def predict(message, history):
    user_query = message

    relevant_chunks = get_relevant_chunks(vector_db, user_query, 3)



    prompt = f"""
        User query:
        {user_query}

        Context:
        {relevant_chunks}
    """

    # Step 2.3 Send the prompt to the LLM and get the final response
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.6
    )

    print(f"Prompt: {prompt}")

    messages.append(("human", prompt)) #sempre a 1 system, que controla a lógica da conversa, e depois vai ter human assistant human assistant
    response = llm.invoke(messages)
    print(response)

    messages.append(("assistant", response.content))
    print(messages)
    return response.content


# Load the information and prepare the RAG
# load_information()

gr.ChatInterface(predict, type="messages").launch(debug=True)
