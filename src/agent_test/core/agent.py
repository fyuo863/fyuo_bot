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
class UserInputRequired:
    """需要用户输入事件 (let_user_answer 触发)"""
    question: str
    call_id: str


@dataclass
class AgentComplete:
    """Agent 任务完成"""
    pass


class ReActAgent:
    """ReAct Agent 核心循环 —— 纯逻辑，不含任何显示/IO。

    用法:
        agent = ReActAgent(tools=[...], system="...")
        for event in agent.run(user_input):
            match event:
                case TextChunk():      ...
                case ReasoningChunk(): ...
                case ToolCall():       ...
                case ToolResult():     ...
                case UserInputRequired(): ...  # 调用方负责获取用户输入
                case AgentComplete():  ...
    """

    def __init__(
        self,
        tools: list[BaseTool],
        system: str = "",
    ):
        self.tool_registry: dict[str, BaseTool] = {t.name: t for t in tools}
        self.system = system
        self.messages: list[dict] = []

    def _tool_schemas(self) -> list[dict]:
        return [t.to_openai_schema() for t in self.tool_registry.values()]

    def run(self, user_input: str, max_iterations: int = 20):
        """ReAct: Think → Act → Observe 循环生成器。"""
        self.messages = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": user_input},
        ]

        for _ in range(max_iterations):
            # ===== Step 1: Think =====
            stream_items: list = []
            for item in AgentChat.chat(
                messages=self.messages,
                tools=self._tool_schemas(),
            ):
                stream_items.append(item)
                yield item

            assistant_msg = Usage.assemble_assistant_message(stream_items)
            self.messages.append(assistant_msg)

            # 无 tool_calls → 任务完成
            if "tool_calls" not in assistant_msg:
                yield AgentComplete()
                return

            # ===== Step 2+3: Act + Observe =====
            for tc in assistant_msg["tool_calls"]:
                func_name = tc["function"]["name"]
                args = json.loads(tc["function"]["arguments"])
                tool = self.tool_registry.get(func_name)

                if tool is None:
                    result = f"错误：未找到工具 '{func_name}'"
                elif func_name == "let_user_answer":
                    yield UserInputRequired(
                        question=args.get("question", ""),
                        call_id=tc["id"],
                    )
                    return  # 暂停，等待调用方喂入用户回复
                else:
                    result = tool.execute(**args)

                if func_name != "let_user_answer":
                    yield ToolResult(name=func_name, result=result)
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": str(result),
                    })

        yield AgentComplete()

    def feed_user_response(self, user_response: str, call_id: str):
        """调用方收到 UserInputRequired 后，将用户输入喂回 agent。"""
        self.messages.append({
            "role": "tool",
            "tool_call_id": call_id,
            "content": "用户已回复",
        })
        self.messages.append({
            "role": "user",
            "content": user_response,
        })

    def resume(self, max_iterations: int = 20):
        """喂入用户回复后，继续 ReAct 循环（复用当前 messages）。"""
        for _ in range(max_iterations):
            stream_items: list = []
            for item in AgentChat.chat(
                messages=self.messages,
                tools=self._tool_schemas(),
            ):
                stream_items.append(item)
                yield item

            assistant_msg = Usage.assemble_assistant_message(stream_items)
            self.messages.append(assistant_msg)

            if "tool_calls" not in assistant_msg:
                yield AgentComplete()
                return

            for tc in assistant_msg["tool_calls"]:
                func_name = tc["function"]["name"]
                args = json.loads(tc["function"]["arguments"])
                tool = self.tool_registry.get(func_name)

                if tool is None:
                    result = f"错误：未找到工具 '{func_name}'"
                elif func_name == "let_user_answer":
                    yield UserInputRequired(
                        question=args.get("question", ""),
                        call_id=tc["id"],
                    )
                    return
                else:
                    result = tool.execute(**args)

                yield ToolResult(name=func_name, result=result)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": str(result),
                })

        yield AgentComplete()
