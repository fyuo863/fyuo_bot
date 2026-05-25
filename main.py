import os
from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("OPENAI_MODEL_NAME", "deepseek-v4-pro")
base_url = os.getenv("OPENAI_BASE_URL")

def chat(prompt: str, model: str = "deepseek-v4-pro", system: str | None = None):
    """注意：这里不再直接 return 一个字符串，而是变成了一个持续产出数据的生成器"""
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    # 开启流式
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        stream=True, 
    )
    
    # 【关键修改】遍历 Stream 对象，用 yield 逐个吐出文字块
    for chunk in response:
        delta_content = chunk.choices[0].delta.content
        if delta_content is not None:
            yield delta_content

def main():
    print("AI: ", end="", flush=True)
    
    # 【关键修改】接收端也必须用 for 循环，一边接收一边打印
    # 不能再写 reply = chat(...) 了！
    for chunk in chat("用一句话介绍 Python"):
        print(chunk, end="", flush=True)
        
    print() # 打印完毕后换行

if __name__ == "__main__":
    main()