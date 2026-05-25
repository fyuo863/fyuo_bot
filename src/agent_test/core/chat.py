from openai import OpenAI
from config import config


class AgentChat:
    def __init__(self, agent):
        self.agent = agent

    def send_message(self, message):
        response = self.agent.process_message(message)
        return response

    @staticmethod
    def chat(prompt: str, system: str | None = None, stream: bool = True):
        """注意：这里不再直接 return 一个字符串，而是变成了一个持续产出数据的生成器"""
        client = OpenAI(api_key = config.api_key, base_url = config.base_url)
        
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # 开启流式
        response = client.chat.completions.create(
            model=config.model_name,
            messages=messages,
            temperature=0.7,
            stream=stream, 
        )
        if stream:
            # 【关键修改】遍历 Stream 对象，用 yield 逐个吐出文字块
            for chunk in response:
                delta_content = chunk.choices[0].delta.content
                if delta_content is not None:
                    yield delta_content
        else:        # 如果不使用流式，则直接返回完整回复
            yield response.choices[0].message.content