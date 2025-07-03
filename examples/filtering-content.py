import os
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))
# Now adding docs with metadata (in this case, the page number)
list_of_documents = [
    Document(page_content="foo", metadata=dict(page=1)),
    Document(page_content="bar", metadata=dict(page=1)),
    Document(page_content="123123125412512", metadata=dict(page=1)),
    Document(page_content="1231231sefsefse", metadata=dict(page=1)),
    Document(page_content="123", metadata=dict(page=1)),
    Document(page_content="324236", metadata=dict(page=1)),
    Document(page_content="foo", metadata=dict(page=2)),
    Document(page_content="barbar", metadata=dict(page=2)),
    Document(page_content="foo", metadata=dict(page=3)),
    Document(page_content="bar burr", metadata=dict(page=3)),
    Document(page_content="foo", metadata=dict(page=4, access="ceo")),
    Document(page_content="bar bruh", metadata=dict(page=4, access="dev")),
]
db = FAISS.from_documents(list_of_documents, embeddings)

results_with_scores = db.similarity_search_with_score("foo", k=4)
for doc, score in results_with_scores:
    print(f"Content: {doc.page_content}, Metadata: {doc.metadata}, Score: {score}")

# Now filtering by page
results_with_scores = db.similarity_search_with_score("foo", filter=dict(page=4, access="dev"), k=4)
for doc, score in results_with_scores:
    print(f"Content: {doc.page_content}, Metadata: {doc.metadata}, Score: {score}")