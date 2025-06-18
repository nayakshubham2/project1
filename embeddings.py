import os
from openai import OpenAI
import time
import numpy as np


client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))


def get_openai_embedding(text: str, model: str = "text-embedding-3-small") -> list[float]:
    """
    Generates an embedding for a given text using OpenAI's API.
    Handles retries for transient network errors.
    """
    # OpenAI's API recommends replacing newlines to avoid issues.
    text = text.replace("\n", " ")
    
    for i in range(3): # Retry up to 3 times
        try:
            response = client.embeddings.create(
                input=[text], # API expects a list of texts
                model=model
            )
            # The actual embedding vector is in response.data[0].embedding
            return response.data[0].embedding
        except Exception as e:
            print(f"An error occurred with the OpenAI API: {e}")
            if i < 2:
                print("Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print("Failed to get embedding after multiple retries.")
                return None # Return None on failure


# Load Embedddings
def load_embeddings():
    data = np.load("tds_database.npz", allow_pickle=True)
    data['vectors'], data['metadata']
    
    return data['vectors'], data['metadata']