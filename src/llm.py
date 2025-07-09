# LLM
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.6
)

default_system_prompt = f"""
    Instructions:
    Answer the provided user query
    Use the provided context, chat history, or both to deduce the answer.
    The chat history is ordered from the first to the latest interaction, the user is a message from the user, the assistant is a message from the LLM.
    If you don't know the answer, say 'I don't know'
"""

messages = [
    ("system", default_system_prompt)
]

# Query the LLM with the necessary content:
def query_llm(user_query, relevant_chunks):
    prompt = f"""
        User query:
        {user_query}

        Context:
        {relevant_chunks}
    """

    messages.append(("human", prompt))

    response = llm.invoke(messages)

    messages.append(("assistant", response.content))

    return response.content