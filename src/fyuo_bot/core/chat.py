from dataclasses import dataclass
from openai import OpenAI
from config.config import api_key, base_url, model_name

import tiktoken

# DeepSeek 系列模型使用类似 cl100k_base 的 BPE 分词器
_tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(messages: list[dict]) -> int:
    """估算消息列表的 token 数。"""
    total = 0
    for msg in messages:
        total += len(_tokenizer.encode(msg.get("content", "")))
        total += len(_tokenizer.encode(msg.get("role", "")))
        # tool_calls 的粗略估算
        for tc in msg.get("tool_calls", []):
            total += len(_tokenizer.encode(str(tc)))
        # reasoning_content 也计入
        if msg.get("reasoning_content"):
            total += len(_tokenizer.encode(msg["reasoning_content"]))
    return total


@dataclass
class TextChunk:
    content: str


@dataclass
class ToolCall:
    call_id: str
    name: str
    arguments: str  # JSON 字符串

@dataclass
class ReasoningChunk:
    """深度思考模型的推理过程片段"""
    content: str

class AgentChat:
    def __init__(self, agent):
        self.agent = agent

    def send_message(self, message):
        response = self.agent.process_message(message)
        return response



    @staticmethod
    def chat(
        messages: list[dict],
        stream: bool = True,
        tools: list[dict] | None = None,
        model: str | None = None,
    ):
        """流式对话生成器，支持 tool calling。

        Yields:
            TextChunk: 文本片段
            ToolCall:  模型请求调用工具的意图（仅在 tools 参数传入时可能产生）
        """
        client = OpenAI(api_key=api_key, base_url=base_url)

        kwargs = dict(
            model=model or model_name,
            messages=messages,
            temperature=0.7,
            stream=stream,
        )
        
        if tools:
            kwargs["tools"] = tools

        response = client.chat.completions.create(**kwargs)

        # ========== 以下您的流式解析逻辑写得非常棒，完全保持原样！ ==========
        if stream:
            tool_call_buf: dict[int, dict] = {}  # index -> {id, name, arguments}
            for chunk in response:
                delta = chunk.choices[0].delta

                # 【新增】捕获深度思考模型的推理过程
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning is not None:
                    yield ReasoningChunk(content=reasoning)

                # 文本内容
                if delta.content is not None:
                    yield TextChunk(content=delta.content)

                # tool call 增量
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_call_buf:
                            tool_call_buf[idx] = {"call_id": "", "name": "", "arguments": ""}
                        entry = tool_call_buf[idx]
                        if tc.id:
                            entry["call_id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                entry["name"] = tc.function.name
                            if tc.function.arguments:
                                entry["arguments"] += tc.function.arguments

            # 流结束后，产出完整的 tool call
            for idx in sorted(tool_call_buf.keys()):
                entry = tool_call_buf[idx]
                if entry["name"]:
                    yield ToolCall(
                        call_id=entry["call_id"],
                        name=entry["name"],
                        arguments=entry["arguments"],
                    )
        else:
            message = response.choices[0].message
            if message.content:
                yield TextChunk(content=message.content)
            if message.tool_calls:
                for tc in message.tool_calls:
                    yield ToolCall(
                        call_id=tc.id,
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                    )