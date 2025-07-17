# LLM
from langchain_openai import ChatOpenAI
import json

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.6
)

llm_transformation = ChatOpenAI(
    model="gpt-4.1-nano",
    temperature=0.8
)

default_system_prompt = f"""
    Instructions:
    Answer the provided user query.
    Use the provided context, chat history, or both to deduce the answer.
    The chat history is ordered from the first to the latest interaction, the user is a message from the user, the assistant is a message from the LLM.
    If you don't know the answer, say 'I don't know'.
"""

messages = [
    ("system", default_system_prompt)
]

# Queries LLM to transform the user query into simpler queries
def split_user_query_into_queries(user_query):
    split_user_query_prompt = f"""
        Instruction:
        Deconstruct the provided user query into simple queries.

        User Query:
        {user_query}

        Return a json object with the following schema:
        {{
            queries: [query_one, query_two, ...]
        }}
    """

    llm_bind_json = llm_transformation.bind(response_format={"type": "json_object"})
    splitted_queries = llm_bind_json.invoke(split_user_query_prompt).content

    return json.loads(splitted_queries).get("queries")

# Generates 5 different queries per splitted user query in the queries list
def generate_multiple_queries(queries):
    multiple_queries_prompt = f"""
        Instructions:
        The user query is an array of queries.
        Generate 5 different related queries for each item in the user query array.
        Ensure that each generated query has a question mark at the end.
        All the generated queries should be added into a single array.

        User Query:
        {queries}

        Return a json object with the following schema:
        {{
            queries: ['query_one', 'query_two', ...]
        }}
    """

    llm_bind_json = llm_transformation.bind(response_format={"type": "json_object"})
    generated_queries = llm_bind_json.invoke(multiple_queries_prompt).content

    return json.loads(generated_queries).get("queries")

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