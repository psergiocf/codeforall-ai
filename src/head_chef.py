import gradio as gr
from recipes_loader import load_recipes
from recipes_chunker import chunk_recipes
from recipes_db import create_vector_db
from langchain_openai import ChatOpenAI         # LLM


def predict(message, history):
    ################################################################
    ########## Phase 1: Load information and prepare RAG ###########
    ################################################################
    # Load recipes from JSON file
    recipes = load_recipes()

    # Transform recipes into chunks
    chunks = chunk_recipes(recipes)

    # Embed the chunks and store them in a vector database
    # vector_db = create_vector_db(chunks)



    ################################################################
    ############# Phase 2: RAG Retrieval Augmentation ##############
    ################################################################

    # user_query = message

    # # Step 2.1 - Retrieve dos chunks mais relevantes
    # # Custos
    # relevant_chunks = vector_db.similarity_search(user_query, 3)
    # print(relevant_chunks)

    # # Step 2.2 Create final prompt to send to LLM
    # prompt = f"""
    #     Instructions:
    #     Answer the provided user query with the given context
    #     Only answer questions that the answer is provided in the given context
    #     If you don't know the answer, say 'I don't know'

    #     User query:
    #     {user_query}

    #     Context:
    #     {relevant_chunks}
    # """

    # # Step 2.3 Send the prompt to the LLM and get the final response
    # llm = ChatOpenAI(
    #     model="gpt-4o-mini",
    #     temperature=0.6
    # )

    # print(f"Prompt: {prompt}")

    # # 3 Roles
    # # system/developer - Instruir toda a conversa
    # # assistant/ai - resposta da LLM
    # # human/user - message do utilizador
    # messages = [
    #     ("human", prompt),
    # ]

    # response = llm.invoke(messages)
    # print(response)

    # return response.content

    return "Hi Sérgio, I'm the Head Chef AI!"

# Load the information and prepare the RAG
# vector_db = load_information()

gr.ChatInterface(predict, type="messages").launch(debug=True)
