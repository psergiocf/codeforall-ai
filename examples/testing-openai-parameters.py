#!/usr/bin/env python

import os
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken

# Load environment variables from .env file
load_dotenv()

client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY')
)

encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
encodedText = encoder.encode("I'm from Portugal.")

print(encodedText)
print(encoder.decode(encodedText))

print(len(encodedText))

for tokenId in encodedText:
  print(f"{tokenId} - {encoder.decode([tokenId])}")

def generateWithTemp(temp):
    response = client.completions.create(
        model = "gpt-3.5-turbo-instruct",
        prompt= "Say something about what you think of Portugal",
        max_tokens= 50,
        temperature = temp
    )
    return response.choices[0].text


def generateWithTopP(topP):
    response = client.completions.create(
        model = "gpt-3.5-turbo-instruct",
        prompt= "Say something about what you think of Portugal",
        max_tokens= 50,
        top_p = topP
    )
    return response.choices[0].text

def generateWithFrequencyPenalty(frequencyPenalty):
    response = client.completions.create(
        model = "gpt-3.5-turbo-instruct",
        prompt= "Say something about what you think of Portugal",
        max_tokens= 50,
        frequency_penalty=frequencyPenalty
    )
    return response.choices[0].text

for temp in [0, 0.5, 1, 1.5, 2]:
  # good values - 0.3 -> 0.7
  print(f"Temperature: {temp} - Generated: {generateWithTemp(temp)}\n")

for topP in [0, 0.5, 1]:
  print(f"Top P: {topP} - Generated: {generateWithTopP(topP)}\n")

for frequencyPenalty in [-2, -1, 0, 1, 2]:
  print(f"Frequency Penalty: {frequencyPenalty} - Generated: {generateWithFrequencyPenalty(frequencyPenalty)}\n")