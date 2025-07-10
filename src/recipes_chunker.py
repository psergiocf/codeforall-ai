import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Transform recipes into chunks
# https://python.langchain.com/docs/how_to/#text-splitters
def chunk_recipes(recipes):
    chunk_size = 1000                  # 500 to 1000 (increase/decrease by 100)
    chunk_overlap = chunk_size * 0.2   # 15% to 30% of the chunk size
    recipes_documents = []
    chunks = []

    for recipe in recipes:
        recipe_document = create_recipe_document(recipe)
        recipes_documents.append(recipe_document)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_documents(recipes_documents)
    
    return chunks

def create_recipe_document(recipe):
    recipe_json_page_content = json.loads(recipe.page_content)
    recipe_text = format_recipe(recipe_json_page_content)

    # Create a new document with the recipe text
    return Document(
        page_content=recipe_text,
        metadata={
            **recipe.metadata
        }
    )

# Returns an object with the recipe"s filtering data
def prepare_recipe_metadata(recipe):
    filter = {}

    if "id" in recipe:
        filter["id"] = recipe["id"]

    if "name" in recipe:
        filter["name"] = recipe["name"]

    if "author" in recipe:
        filter["author"] = recipe["author"]

    if "rattings" in recipe:
        filter["rating"] = recipe["rattings"]

    if "ingredients" in recipe:
        filter["ingredients"] = "|".join(recipe["ingredients"])

    if "nutrients" in recipe and "kcal" in recipe["nutrients"]:
        filter["calories"] = recipe["nutrients"]["kcal"]

    if "nutrients" in recipe and "protein" in recipe["nutrients"]:
        filter["protein"] = recipe["nutrients"]["protein"]

    if "serves" in recipe:
        filter["serves"] = recipe["serves"]

    if "difficult" in recipe:
        filter["difficulty"] = recipe["difficult"]

    if "subcategory" in recipe:
        filter["subcategory"] = recipe["subcategory"]

    return filter

# Return new recipes documents with updated metadata
def update_recipes_metadata(recipes):
    new_recipes = []

    for recipe in recipes:
        metadata = prepare_recipe_metadata(json.loads(recipe.page_content))

        new_recipes.append(
            Document(
                page_content=recipe.page_content,
                metadata={
                    **recipe.metadata,
                    **metadata
                }
            )
        )

    return new_recipes
