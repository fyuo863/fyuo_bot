from dataclasses import dataclass
from openai import OpenAI
from config.config import api_key, base_url, model_name


@dataclass
class TextChunk:
    content: str


@dataclass
class ToolCall:
    call_id: str
    name: str
    arguments: str  # JSON 字符串


class AgentChat:
    def __init__(self, agent):
        self.agent = agent

    def send_message(self, message):
        response = self.agent.process_message(message)
        return response

    @staticmethod
    def chat(
        messages: list[dict],  # 【关键修改 1】不再接收单句 prompt 和 system，直接接收完整的上下文列表
        stream: bool = True,
        tools: list[dict] | None = None,
    ):
        """流式对话生成器，支持 tool calling。

        Yields:
            TextChunk: 文本片段
            ToolCall:  模型请求调用工具的意图（仅在 tools 参数传入时可能产生）
        """
        # 注意：请确保 api_key 和 base_url 已经在外部正确获取或导入
        client = OpenAI(api_key=api_key, base_url=base_url)

        # 【关键修改 2】删除了这里原本手动组装 system 和 user 角色的代码
        # 因为在真正的 Agent 循环中，这些角色和历史对话都已经存在于传入的 messages 列表里了

        kwargs = dict(
            model=model_name,
            messages=messages,  # 【关键修改 3】直接将完整的历史列表透传给 API
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