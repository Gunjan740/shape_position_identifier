import pandas as pd
import openai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key from environment variables
# Ensure you have OPENAI_API_KEY set in your .env file
openai.api_key = os.getenv("OPENAI_API_KEY")

def main():
    print("Shape Position Identifier project started!")
    # Example: Check if API key is loaded (optional)
    if openai.api_key:
        print("OpenAI API key loaded successfully.")
    else:
        print("Error: OpenAI API key not found. Make sure it's set in your .env file.")

    # TODO: Add your project logic here

if __name__ == "__main__":
    main()
