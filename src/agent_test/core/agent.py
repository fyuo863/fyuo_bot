import json
from dataclasses import dataclass

from .chat import AgentChat, TextChunk, ReasoningChunk, ToolCall
from .usage import Usage
from tools.base import BaseTool


@dataclass
class ToolResult:
    """工具执行完成事件"""
    name: str
    result: str


@dataclass
class AgentComplete:
    """Agent 任务完成事件"""
    pass


class ReActAgent:
    """ReAct Agent：Think → Act → Observe 循环。

    用法:
        agent = ReActAgent(
            tools=[GetWeatherTool(), GetLocationTool()],
            system="你是一个生活助手。",
        )
        for event in agent.run("今天天气怎么样？"):
            # 按类型分发显示
    """

    def __init__(
        self,
        tools: list[BaseTool],
        system: str = "",
        max_iterations: int = 10,
    ):
        self.tool_registry: dict[str, BaseTool] = {t.name: t for t in tools}
        self.system = system
        self.max_iterations = max_iterations
        self.messages: list[dict] = []

    def _tool_schemas(self) -> list[dict]:
        return [t.to_openai_schema() for t in self.tool_registry.values()]

    def run(self, user_input: str):
        """ReAct 核心循环，逐个产出事件供调用方渲染。

        Yields: TextChunk | ReasoningChunk | ToolCall | ToolResult | AgentComplete
        """
        # —— 初始化上下文 ——
        self.messages = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": user_input},
        ]

        for _ in range(self.max_iterations):
            # ===== Step 1: Think =====
            stream_items: list = []
            for item in AgentChat.chat(
                messages=self.messages,
                tools=self._tool_schemas(),
            ):
                stream_items.append(item)
                yield item  # 实时透传 TextChunk / ReasoningChunk / ToolCall

            # 持久化 assistant 消息
            assistant_msg = Usage.assemble_assistant_message(stream_items)
            self.messages.append(assistant_msg)

            # 没有 tool_calls → 任务完成
            if "tool_calls" not in assistant_msg:
                yield AgentComplete()
                return

            # ===== Step 2+3: Act + Observe =====
            for tc in assistant_msg["tool_calls"]:
                func_name = tc["function"]["name"]
                tool = self.tool_registry.get(func_name)
                if tool is None:
                    result = f"错误：未找到工具 '{func_name}'"
                else:
                    args = json.loads(tc["function"]["arguments"])
                    result = tool.execute(**args)

                yield ToolResult(name=func_name, result=result)

                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": str(result),
                })

        # 超过最大迭代次数
        yield AgentComplete()
