from langchain_community.document_loaders import JSONLoader

# Load the recipes from the JSON files
def load_recipes():
    print("⏳ Loading recipes from JSON file...")
    loader = JSONLoader(
        file_path="docs/shelf/recipes.json",
        jq_schema=".[]",
        text_content=False,
    )

    # Load the documents
    documents = loader.load()

    return documents
