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

# Formats a recipe from JSON to text
def format_recipe(recipe):
    parts = []

    if "id" in recipe:
        parts.append(f"Id: {recipe['id']}")

    if "url" in recipe:
        parts.append(f"URL: {recipe['url']}")

    if "image" in recipe:
        parts.append(f"Image: {recipe['image']}")

    if "name" in recipe:
        parts.append(f"Recipe: {recipe['name']}")

    if "description" in recipe:
        parts.append(f"Description: {recipe['description']}")

    if "author" in recipe:
        parts.append(f"Author: {recipe['author']}")

    if "rattings" in recipe:
        parts.append(f"Rating: {recipe['rattings']}")

    if "ingredients" in recipe:
        parts.append("Ingredients:")
        for ingredient in recipe["ingredients"]:
            parts.append(f"- {ingredient}")

    if "steps" in recipe:
        parts.append("Steps:")
        for step in recipe["steps"]:
            parts.append(f"- {step}")

    if "nutrients" in recipe:
        parts.append("Nutrients:")
        nutrients = recipe["nutrients"]

        for key, value in nutrients.items():
            parts.append(f"- {key}: {value}")

    if "times" in recipe:
        parts.append("Times:")
        times = recipe["times"]

        for key, value in times.items():
            parts.append(f"- {key}: {value}")

    if "serves" in recipe:
        parts.append(f"Serves: {recipe['serves']}")

    if "difficult" in recipe:
        parts.append(f"Difficulty: {recipe['difficult']}")

    if "vote_count" in recipe:
        parts.append(f"Voting: {recipe['vote_count']}")

    if "subcategory" in recipe:
        parts.append(f"Subcategory: {recipe['subcategory']}")

    if "dish_type" in recipe:
        parts.append(f"Dish Type: {recipe['dish_type']}")

    if "maincategory" in recipe:
        parts.append(f"Main Category: {recipe['maincategory']}")
    
    return "\n".join(parts)