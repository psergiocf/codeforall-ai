import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

def fix_typos(page_content):
    return page_content.replace("rattings", "ratings")

# Returns an object with the recipe"s filtering data
def prepare_recipe_metadata(index, recipe):
    filter = {}

    filter["index"] = index

    if "id" in recipe:
        filter["id"] = recipe["id"]

    if "name" in recipe:
        filter["name"] = recipe["name"]

    if "author" in recipe:
        filter["author"] = recipe["author"]

    if "ratings" in recipe:
        filter["rating"] = recipe["ratings"]

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
def update_recipes_data(recipes):
    new_recipes = []

    for index, recipe in enumerate(recipes):
        initial_page_content = recipe.page_content

        updated_page_content = fix_typos(initial_page_content)
        metadata = prepare_recipe_metadata(index, json.loads(updated_page_content))

        new_recipes.append(
            Document(
                page_content=updated_page_content,
                metadata={
                    **recipe.metadata,
                    **metadata
                }
            )
        )

    return new_recipes
