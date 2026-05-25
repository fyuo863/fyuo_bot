import os
from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("OPENAI_MODEL_NAME", "deepseek-v4-pro")
base_url = os.getenv("OPENAI_BASE_URL")

def chat(prompt: str, model: str = "deepseek-v4-pro", system: str | None = None) -> str:
    """发送一次对话请求，返回模型回复文本。"""
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),  # 可选，使用第三方代理时设置
    )
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
    )
    return response.choices[0].message.content


def main():
    reply = chat("用一句话介绍 Python")
    print(reply)


if __name__ == "__main__":
    main()
