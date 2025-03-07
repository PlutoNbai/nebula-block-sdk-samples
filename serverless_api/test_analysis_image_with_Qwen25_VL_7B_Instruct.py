import os
import requests
import dotenv
import logging

def get_nebula_response(api_key: str, messages: list):
    """Fetch response from Nebula Inference API for chat completion.

    Args:
        api_key: API key for authentication.
        messages: List of message inputs including text and image URLs.

    Returns:
        JSON response from the API.
    """
    url = "https://inference.nebulablock.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "messages": messages,
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "max_tokens": None,
        "temperature": 1,
        "top_p": 0.9,
        "stream": False
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()

def verify_connection(api_key: str):
    """Verify API connection by sending a test request."""
    test_messages = [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"}},
            {"type": "text", "text": "What is this image?"}
        ]}
    ]
    response = get_nebula_response(api_key, test_messages)
    return response is not None

if __name__ == '__main__':
    dotenv.load_dotenv()
    nebula_api_key = os.getenv("NEBULA_API_KEY")

    if not nebula_api_key:
        logging.error("NEBULA_API_KEY is missing.")
    else:
        logging.info("Attempting connection to Nebula Chat Completion API...")
        if verify_connection(nebula_api_key):
            logging.info("Connected to Nebula Chat Completion API successfully.")
            messages = [
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"}},
                    {"type": "text", "text": "What is this image?"}
                ]}
            ]
            response = get_nebula_response(nebula_api_key, messages)
            logging.info(f"API Response: {response}")
        else:
            logging.error("Failed to connect to Nebula Chat Completion API.")
