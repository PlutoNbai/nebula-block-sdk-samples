import os
import requests
import dotenv
import logging

def get_nebula_response(api_key: str, input_texts: list):
    """Fetch embeddings from Nebula Inference API.

    Args:
        api_key: API key for authentication.
        input_texts: List of texts to generate embeddings for.

    Returns:
        JSON response from the API.
    """
    url = "https://inference.nebulablock.com/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": "BAAI/bge-large-en-v1.5",
        "input": input_texts
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()

def verify_connection(api_key: str):
    """Verify API connection by sending a test request."""
    test_inputs = [
        "Bananas are berries, but strawberries are not, according to botanical classifications.",
        "The Eiffel Tower in Paris was originally intended to be a temporary structure."
    ]
    response = get_nebula_response(api_key, test_inputs)
    return response is not None

if __name__ == '__main__':
    dotenv.load_dotenv()
    nebula_api_key = os.getenv("NEBULA_API_KEY")

    if not nebula_api_key:
        logging.error("NEBULA_API_KEY is missing.")
    else:
        logging.info("Attempting connection to Nebula Embeddings API...")
        if verify_connection(nebula_api_key):
            logging.info("Connected to Nebula Embeddings API successfully.")
            input_texts = [
                "Bananas are berries, but strawberries are not, according to botanical classifications.",
                "The Eiffel Tower in Paris was originally intended to be a temporary structure."
            ]
            response = get_nebula_response(nebula_api_key, input_texts)
            logging.info(f"API Response: {response}")
        else:
            logging.error("Failed to connect to Nebula Embeddings API.")
