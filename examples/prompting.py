#!/usr/bin/env python

import os
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken

# Load environment variables from .env file
load_dotenv()

client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
)

encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
encodedText = encoder.encode("I'm from Portugal.")

print(encodedText)
print(encoder.decode(encodedText))

print(len(encodedText))

for tokenId in encodedText:
  print(f"{tokenId} - {encoder.decode([tokenId])}")

prompt_a = "Write me a funny joke"
prompt_b = "Write me a funny joke about programming"
prompt_c = "Write me a funny joke about programming, make it a knock knock joke"
prompt_d = "Write me a funny joke about programming, make it a knock knock joke. For context this is a good joke “Knock, knock.” “Who’s there?” very long pause…. “Java.”"

def generateWithTemp(prompt):
    response = client.completions.create(
        model = "gpt-3.5-turbo-instruct",
        prompt= prompt,
        max_tokens= 50
    )
    return response.choices[0].text

for prompt in [prompt_a, prompt_b, prompt_c, prompt_d]:
  print(f"Prompt: {prompt} - Generated: {generateWithTemp(prompt)}\n")