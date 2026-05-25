import os

api_key = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("OPENAI_MODEL_NAME", "deepseek-v4-pro")
base_url = os.getenv("OPENAI_BASE_URL")
