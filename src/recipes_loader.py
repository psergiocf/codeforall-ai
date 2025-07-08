# import json
from langchain_community.document_loaders import JSONLoader

# Load the recipes from the JSON files
def load_recipes():
    print("Loading recipes from JSON file...")
    loader = JSONLoader(
        file_path="docs/shelf/recipes.json",
        jq_schema=".[]",
        text_content=False,
    )

    # Load the documents
    documents = loader.load()
    print(f"Loaded documents: {len(documents)}")
    # print(f"Documents: {documents}")

    # for document in documents:
    #     page_content = json.loads(document.page_content)
    #     doc_id = page_content.get('id')
    #     print(f"Document: {document}")
    #     print("---")
    #     print(f"ID: {doc_id}")
    #     print("---------------------------")

    return documents




#################### KaggleHub Example ##########################
#import kagglehub

# Download latest version
#path = kagglehub.dataset_download("crispen5gar/recipes3k")

#print(f"Dataset downloaded to: {path}")
##################################################################