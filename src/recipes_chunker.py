import json
from langchain_text_splitters import RecursiveJsonSplitter

# Transform documents into chunks
# https://python.langchain.com/docs/how_to/#text-splitters
def chunk_recipes(documents):
    chunk_size = 500 # 500 to 1000 (increase/decrease by 100)
    json_splitter = RecursiveJsonSplitter(chunk_size)
    chunks = []

    for document in documents:
        page_content = json.loads(document.page_content)
        document_chunks = json_splitter.split_json(page_content)
        chunks.extend(document_chunks)

    print(f"All chunks: {chunks}")

    return chunks