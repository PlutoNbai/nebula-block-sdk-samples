import os
import requests
import dotenv
import logging

def get_nebula_response(api_key: str, prompt: str):
    """Fetch response from Nebula Inference API.

    Args:
        api_key: API key for authentication.
        prompt: User's query for the AI model.

    Returns:
        JSON response from the API.
    """
    url = "https://api.nebulablock.com/api/v1/images/generation"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model_name": "stabilityai/stable-diffusion-xl-base-1.0",
        "prompt": prompt,
        "num_steps": 25,
        "guidance_scale": 9,
        "negative_prompt": None,
        "width": 1024,
        "height": 1024
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()

def verify_connection(api_key: str):
    """Verify API connection by sending a test request."""
    test_prompt = "Test image generation connectivity."
    response = get_nebula_response(api_key, test_prompt)
    return response is not None

if __name__ == '__main__':
    dotenv.load_dotenv()
    nebula_api_key = os.getenv("NEBULA_API_KEY")

    if not nebula_api_key:
        logging.error("NEBULA_API_KEY is missing.")
    else:
        logging.info("Attempting connection to Nebula Image Generation API...")
        if verify_connection(nebula_api_key):
            logging.info("Connected to Nebula Image Generation API successfully.")
            user_prompt = "A futuristic cityscape with neon lights."
            response = get_nebula_response(nebula_api_key, user_prompt)
            logging.info(f"API Response: {response}")
        else:
            logging.error("Failed to connect to Nebula Image Generation API.")
