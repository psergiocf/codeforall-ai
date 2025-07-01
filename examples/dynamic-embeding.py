import os
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

text = "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum."
chunk_size=50
chunk_overlap = 50 * 0.2 # 15% - 30%
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
chunk_strings = [chunk.page_content for chunk in text_splitter.split_text(text)]

embeddings_model = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'), model="text-embedding-3-small")

embeddings = embeddings_model.embed_documents(chunk_strings)
print(len(embeddings), len(embeddings[0]))

print(embeddings[:2])